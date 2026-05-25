"""通用工具：上传路径、检测图命名、letterbox 预处理。"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# --- upload paths ---

def _norm_rel(path: str) -> str:
    return str(path).replace('\\', '/').lstrip('./')


def to_upload_relative_path(path, upload_root) -> str | None:
    """
    将绝对路径或 /api/files/ URL 转为相对 UPLOAD_FOLDER 的路径（正斜杠）。
    已在 upload 目录外的绝对路径返回 None。
    """
    if not path:
        return None
    s = str(path).strip()
    if not s or s.startswith(('http://', 'https://')):
        return None
    if s.startswith('/api/files/'):
        return _norm_rel(s[len('/api/files/') :])
    if s.startswith('/uploads/'):
        return _norm_rel(s.replace('/uploads/', '', 1))

    root = os.path.normpath(str(upload_root))
    if os.path.isabs(s):
        try:
            rel = os.path.relpath(os.path.normpath(s), root)
        except ValueError:
            return None
        if rel.startswith('..'):
            return None
        return _norm_rel(rel)
    return _norm_rel(s)


def resolve_stored_file_path(stored, upload_root):
    """将库中路径或 /api/files/、/uploads/ 风格 URL 解析为绝对磁盘路径。"""
    if not stored:
        return None
    rel = to_upload_relative_path(stored, upload_root)
    if not rel:
        return None
    return os.path.normpath(os.path.join(upload_root, rel.replace('/', os.sep)))


def safe_remove_file(abs_path):
    if not abs_path:
        return
    try:
        p = os.path.normpath(abs_path)
        if os.path.isfile(p):
            os.remove(p)
    except OSError:
        pass


# --- image naming ---

# 与检测、头像、数据集写入等共用
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

_FILENAME_FORBIDDEN = frozenset('<>:"/\\|?*')


def sanitize_username_for_filename(username: str | None, max_len: int = 40) -> str:
    """
    将登录名转为可安全用于文件名的片段；保留中文、字母、数字及常见符号，去掉路径非法字符。
    无法得到有效片段时返回空串（调用方回退为不含用户名的旧式命名）。
    """
    if username is None:
        return ''
    s = str(username).strip()
    if not s or s in ('.', '..'):
        return ''
    out: list[str] = []
    for ch in s:
        o = ord(ch)
        if o < 32:
            out.append('_')
        elif ch in _FILENAME_FORBIDDEN:
            out.append('_')
        else:
            out.append(ch)
    s = ''.join(out).strip().rstrip('. ')
    if not s or s in ('.', '..'):
        return ''
    if len(s) > max_len:
        s = s[:max_len].rstrip('. ')
    return s


def detect_style_image_basename(
    user_id: int,
    original_filename: str,
    seq: int = 1,
    ts_ms: int | None = None,
    username: str | None = None,
    model_version: str | None = None,
) -> str:
    """
    生成图片文件名（不含路径）。
    ts_ms：可选；同一批多图（如一次检测多框入库）可传入相同时间戳，仅用 seq 区分。
    username：可选；有则插入清理后的用户名片段（支持中文用户名）。
    """
    if ts_ms is None:
        ts_ms = int(time.time() * 1000)
    name = original_filename or ''
    ext = name.rsplit('.', 1)[-1].lower() if '.' in name else 'jpg'
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        ext = 'jpg'
    if ext == 'jpeg':
        ext = 'jpg'
    uid = int(user_id)
    sid = int(seq)
    ver = str(model_version or '').strip()
    ver_part = ''
    if ver:
        lower = ver.lower()
        if lower.endswith(('.pt', '.pth')):
            stem = Path(ver).stem or ver
            ver_part = f'_{stem}'
        elif ver.lower().startswith('v') and ver[1:].isdigit():
            ver_part = f'_{ver}'
        else:
            ver_part = f'_{ver}'
    slug = sanitize_username_for_filename(username)
    if slug:
        return f'{uid}_{slug}_{int(ts_ms)}_detect_{sid}{ver_part}.{ext}'
    return f'{uid}_{int(ts_ms)}_detect_{sid}{ver_part}.{ext}'


# --- image preprocess ---

# 检测接口上传单图上限（字节）
MAX_DETECT_IMAGE_BYTES = 20 * 1024 * 1024
DETECT_IMAGE_BOX_SIZE = 640
# YOLO 常用灰底 padding
LETTERBOX_PAD_COLOR = (114, 114, 114)


class ImageUploadError(ValueError):
    """用户上传图片不合法（大小、格式等）。"""


@dataclass(frozen=True)
class LetterboxMeta:
    """原图 → letterbox 的变换参数，用于将推理框映射回原图坐标。"""

    orig_w: int
    orig_h: int
    scale: float
    pad_left: int
    pad_top: int
    nw: int
    nh: int
    box_size: int

    @classmethod
    def from_shape(cls, h: int, w: int, box_size: int = DETECT_IMAGE_BOX_SIZE) -> LetterboxMeta:
        if h < 1 or w < 1:
            raise ImageUploadError('无效的图片尺寸')
        scale = min(box_size / h, box_size / w)
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        pad_top = (box_size - nh) // 2
        pad_left = (box_size - nw) // 2
        return cls(
            orig_w=int(w),
            orig_h=int(h),
            scale=float(scale),
            pad_left=int(pad_left),
            pad_top=int(pad_top),
            nw=int(nw),
            nh=int(nh),
            box_size=int(box_size),
        )


def letterbox_to_square_bgr(
    image: np.ndarray, size: int = DETECT_IMAGE_BOX_SIZE
) -> tuple[np.ndarray, LetterboxMeta]:
    """等比例缩放至不超过 size×size，再居中填充为 size×size。"""
    h, w = image.shape[:2]
    meta = LetterboxMeta.from_shape(h, w, size)
    nh, nw = meta.nh, meta.nw
    interp = cv2.INTER_AREA if meta.scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (nw, nh), interpolation=interp)
    pad_bottom = size - nh - meta.pad_top
    pad_right = size - nw - meta.pad_left
    out = cv2.copyMakeBorder(
        resized,
        meta.pad_top,
        pad_bottom,
        meta.pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=LETTERBOX_PAD_COLOR,
    )
    return out, meta


def map_bbox_letterbox_to_orig(bbox: list[float], meta: LetterboxMeta) -> list[float]:
    """将 letterbox 图上的 xyxy 映射回原图像素坐标。"""
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    inv = 1.0 / meta.scale if meta.scale > 0 else 1.0
    x1 = (x1 - meta.pad_left) * inv
    y1 = (y1 - meta.pad_top) * inv
    x2 = (x2 - meta.pad_left) * inv
    y2 = (y2 - meta.pad_top) * inv
    x1 = max(0.0, min(float(meta.orig_w), x1))
    y1 = max(0.0, min(float(meta.orig_h), y1))
    x2 = max(0.0, min(float(meta.orig_w), x2))
    y2 = max(0.0, min(float(meta.orig_h), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def map_detections_to_orig(
    detections: list[dict[str, Any]], meta: LetterboxMeta
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in detections or []:
        item = dict(d)
        bbox = item.get('bbox')
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            item['bbox'] = map_bbox_letterbox_to_orig(list(bbox[:4]), meta)
        out.append(item)
    return out


def _read_upload_bytes(file_storage, max_bytes: int) -> bytes:
    stream = getattr(file_storage, 'stream', None) or file_storage
    try:
        stream.seek(0)
    except (AttributeError, OSError):
        pass
    raw = stream.read()
    if not raw:
        raise ImageUploadError('文件为空')
    if len(raw) > max_bytes:
        mb = max(1, max_bytes // (1024 * 1024))
        raise ImageUploadError(f'图片大小不能超过 {mb}MB')
    return raw


def decode_upload_bgr(file_storage, *, max_bytes: int = MAX_DETECT_IMAGE_BYTES) -> tuple[np.ndarray, bytes]:
    """读取上传文件为 BGR 数组，并返回原始字节（用于原图落盘）。"""
    raw = _read_upload_bytes(file_storage, max_bytes)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageUploadError('无法读取图片，请使用 jpg、png、webp 等常见格式')
    return img, raw


def prepare_detect_upload(
    file_storage,
    original_dest: str | Path,
    *,
    max_bytes: int = MAX_DETECT_IMAGE_BYTES,
    box_size: int = DETECT_IMAGE_BOX_SIZE,
) -> tuple[LetterboxMeta, np.ndarray]:
    """
    原图按上传字节原样写入 original_dest；返回 letterbox 推理图与变换 meta。
    """
    img, raw = decode_upload_bgr(file_storage, max_bytes=max_bytes)
    dest = Path(original_dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)

    letterbox_bgr, meta = letterbox_to_square_bgr(img, box_size)
    return meta, letterbox_bgr


def prepare_detect_inference_only(
    file_storage,
    *,
    max_bytes: int = MAX_DETECT_IMAGE_BYTES,
    box_size: int = DETECT_IMAGE_BOX_SIZE,
) -> tuple[LetterboxMeta, np.ndarray]:
    """预览/实时帧：不落盘原图，仅解码并 letterbox。"""
    img, _raw = decode_upload_bgr(file_storage, max_bytes=max_bytes)
    letterbox_bgr, meta = letterbox_to_square_bgr(img, box_size)
    return meta, letterbox_bgr

