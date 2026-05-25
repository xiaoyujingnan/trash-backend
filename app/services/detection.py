"""在线检测：权重、版本、YOLO 推理、检测后样本分流。"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
from ultralytics import YOLO

from app import db
from app.models import PendingSample, User
from app.services.dataset import set_detection_dataset_paths, try_finalize_detection_dataset
from app.utils import LetterboxMeta, map_detections_to_orig

# --- weights ---

_DETECT_MODEL_PREFIX = 'detect_model/'

logger = logging.getLogger(__name__)

_yolo_cache: dict[str, YOLO] = {}


def _configure_torch_threads() -> None:
    try:
        import torch

        n = max(1, min(4, int(os.getenv('TORCH_NUM_THREADS', '2'))))
        torch.set_num_threads(n)
    except Exception:
        pass


_configure_torch_threads()


def _get_yolo(weights_path: str) -> YOLO:
    path = str(Path(weights_path).resolve())
    if path not in _yolo_cache:
        _yolo_cache[path] = YOLO(path)
    return _yolo_cache[path]


def _warmup_yolo(weights_path: str, *, imgsz: int = 640) -> None:
    """启动时预热，避免首次用户检测承担模型加载 + 编译开销。"""
    try:
        import numpy as np

        model = _get_yolo(weights_path)
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        model.predict(dummy, imgsz=imgsz, verbose=False, device='cpu')
        logger.info('detection model warmup finished (%s, imgsz=%s)', weights_path, imgsz)
    except Exception:
        logger.warning('detection model warmup failed', exc_info=True)


def is_usable_weights_file(path: str | Path) -> bool:
    p = Path(path).resolve()
    if not p.is_file() or p.suffix.lower() not in ('.pt', '.pth'):
        return False
    try:
        if p.stat().st_size < 1024:
            return False
    except OSError:
        return False
    try:
        _get_yolo(str(p))
        return True
    except Exception:
        return False


def resolve_detect_model_rel(
    upload_folder: str,
    rel: str,
    *,
    check_usable: bool = True,
) -> Optional[str]:
    rel = str(rel or '').strip().replace('\\', '/').lstrip('/')
    if not rel or '..' in rel or not rel.startswith(_DETECT_MODEL_PREFIX):
        return None
    upload_root = Path(upload_folder).resolve()
    full = (upload_root / rel).resolve()
    try:
        full.relative_to(upload_root)
    except ValueError:
        return None
    if not full.is_file() or full.suffix.lower() not in ('.pt', '.pth'):
        return None
    if check_usable and not is_usable_weights_file(full):
        return None
    return rel.replace('\\', '/')


# --- model version ---

MODEL_VERSION_REL = 'detect_model/model_version.json'


def model_version_config_path(upload_folder: str) -> Path:
    return Path(upload_folder).resolve() / 'detect_model' / 'model_version.json'


def _ensure_model_version_config(upload_folder: str) -> Path:
    """确保 detect_model/model_version.json 存在。"""
    cfg_path = model_version_config_path(upload_folder)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    return cfg_path


def _now_str() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _default_doc() -> dict[str, Any]:
    return {
        'current_model': '',
        'model_path': '',
        'update_time': '',
    }


def _read_doc(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return _default_doc()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _default_doc()
        out = _default_doc()
        out.update(data)
        out.pop('pending_path', None)
        out.pop('version_counter', None)
        return out
    except (OSError, json.JSONDecodeError, TypeError):
        return _default_doc()


def _write_doc(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


def _model_label_from_rel(rel: str) -> str:
    """从 detect_model/<文件名> 得到展示名（保留扩展名，如 111.pt）。"""
    rel = str(rel or '').strip().replace('\\', '/')
    if not rel:
        return ''
    return Path(rel).name


def _resolve_current_model_label(doc: dict[str, Any], ok_rel: str) -> str:
    """current_model 使用权重文件名；兼容旧版 vN 标记。"""
    label = _model_label_from_rel(ok_rel)
    if label:
        return label
    rel = str(doc.get('model_path') or '').strip().replace('\\', '/')
    label = _model_label_from_rel(rel)
    if label:
        return label
    return str(doc.get('current_model') or '').strip()


def get_model_version_info(upload_folder: str) -> dict[str, Any]:
    """读取当前版本元数据及权重路径是否可用。"""
    cfg_path = _ensure_model_version_config(upload_folder)
    doc = _read_doc(cfg_path)
    upload_root = Path(upload_folder).resolve()
    rel = str(doc.get('model_path') or '').strip().replace('\\', '/')
    ok_rel = resolve_detect_model_rel(str(upload_root), rel, check_usable=True) if rel else ''
    label = _resolve_current_model_label(doc, ok_rel)
    stored = str(doc.get('current_model') or '').strip()
    if ok_rel and label and label != stored:
        doc['current_model'] = label
        _write_doc(cfg_path, doc)
    return {
        'current_model': label,
        'model_path': ok_rel or rel,
        'model_path_resolved': ok_rel,
        'update_time': str(doc.get('update_time') or '').strip(),
    }


def read_active_model_rel(upload_folder: str) -> str:
    info = get_model_version_info(upload_folder)
    return info.get('model_path_resolved') or ''


def read_current_model_label(upload_folder: str) -> str:
    info = get_model_version_info(upload_folder)
    return str(info.get('current_model') or '').strip()


def _discover_detect_model_rel(upload_root: Path) -> str | None:
    """model_version 未配置时，在 detect_model/ 下找首个可用 .pt。"""
    base = upload_root / 'detect_model'
    if not base.is_dir():
        return None
    candidates = sorted(
        (p for p in base.iterdir() if p.is_file() and p.suffix.lower() in ('.pt', '.pth')),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in candidates:
        rel = f'detect_model/{p.name}'
        ok = resolve_detect_model_rel(str(upload_root), rel, check_usable=True)
        if ok:
            return ok
    return None


def bootstrap_detection_model(app) -> str:
    """
    应用启动时加载检测权重，优先级：
    1) 环境变量 DETECT_MODEL_REL
    2) model_version.json 的 model_path
    3) detect_model/ 目录下最新 .pt
    """
    upload_root = Path(app.config['UPLOAD_FOLDER']).resolve()
    cfg_path = _ensure_model_version_config(str(upload_root))
    (upload_root / 'detect_model').mkdir(parents=True, exist_ok=True)

    doc = _read_doc(cfg_path)
    rel = str(doc.get('model_path') or '').strip().replace('\\', '/')
    env_rel = str(os.getenv('DETECT_MODEL_REL') or '').strip().replace('\\', '/')
    if env_rel:
        rel = env_rel

    ok = resolve_detect_model_rel(str(upload_root), rel, check_usable=True) if rel else None
    if not ok:
        ok = _discover_detect_model_rel(upload_root)
        if ok:
            doc['model_path'] = ok
            doc['current_model'] = _model_label_from_rel(ok)
            doc['update_time'] = _now_str()
            _write_doc(cfg_path, doc)
            logger.info('detection model auto-selected: %s', ok)

    if ok:
        app.config['ACTIVE_UPLOADED_MODEL_REL'] = ok
        info = get_model_version_info(str(upload_root))
        app.config['CURRENT_MODEL_VERSION'] = info.get('current_model') or ''
        _warmup_yolo(str((upload_root / ok).resolve()))
        logger.info(
            'detection model ready: %s (%s)',
            ok,
            app.config['CURRENT_MODEL_VERSION'] or 'unversioned',
        )
    else:
        app.config['ACTIVE_UPLOADED_MODEL_REL'] = ''
        app.config['CURRENT_MODEL_VERSION'] = ''
        logger.warning(
            'detection model not available at startup (configured: %s); '
            'ensure detect_model/*.pt is deployed or set DETECT_MODEL_REL',
            rel or '(empty)',
        )

    return ok or ''


def _remove_obsolete_model_weights(upload_folder: str, keep_rel: str) -> None:
    """删除 detect_model 下除当前权重外的旧 .pt / .pth 文件。"""
    upload_root = Path(upload_folder).resolve()
    keep_full = (upload_root / keep_rel).resolve()
    base = upload_root / 'detect_model'
    if not base.is_dir():
        return
    for p in base.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in ('.pt', '.pth'):
            continue
        try:
            if p.resolve() == keep_full:
                continue
        except OSError:
            continue
        try:
            p.unlink()
            logger.info('removed obsolete model weight: %s', p)
        except OSError as e:
            logger.warning('failed to remove obsolete model weight %s: %s', p, e)


def activate_detect_model(upload_folder: str, rel: str) -> dict[str, Any]:
    """将 detect_model 下指定权重设为当前全站模型（须可加载），并移除其它旧权重。"""
    ok = resolve_detect_model_rel(upload_folder, rel, check_usable=True)
    if not ok:
        raise ValueError('此模型文件不可用')
    cfg_path = _ensure_model_version_config(upload_folder)
    doc = _read_doc(cfg_path)
    new_label = _model_label_from_rel(ok)
    doc['current_model'] = new_label
    doc['model_path'] = ok
    doc['update_time'] = _now_str()
    _write_doc(cfg_path, doc)
    _remove_obsolete_model_weights(upload_folder, ok)
    return {
        'current_model': new_label,
        'model_path': ok,
        'update_time': doc['update_time'],
    }


# --- service ---

MODEL_UNAVAILABLE_MESSAGE = '当前检测功能不可用'


class ModelUnavailableError(Exception):
    """Raised when the required detection model is not available."""


# 与 datasets/labels/classes.txt 中类别顺序一致（YOLO 类别 id 0..n-1）
DEFAULT_CLASS_NAMES = ['recyclable', 'other', 'hazardous', 'kitchen']


class DetectionService:
    def __init__(self, upload_root=None, active_uploaded_rel=None):
        """
        upload_root: Flask UPLOAD_FOLDER 绝对路径（backend/app/uploads）。
        active_uploaded_rel: 超级管理员激活的全站权重，相对 UPLOAD_FOLDER（detect_model 下 .pt / .pth）。
        """
        if upload_root:
            self.upload_root = Path(upload_root).resolve()
        else:
            self.upload_root = Path(__file__).resolve().parents[2] / 'uploads'
        self.upload_root.mkdir(parents=True, exist_ok=True)

        self._active_uploaded_rel = (active_uploaded_rel or '').strip().replace('\\', '/').lstrip('/')

        env_classes = os.getenv('TRASH_CLASS_NAMES', '').strip()
        if env_classes:
            self.class_names = [c.strip() for c in env_classes.split(',') if c.strip()]
        else:
            self.class_names = list(DEFAULT_CLASS_NAMES)

    def _resolve_weights_path(self, settings=None):
        """仅使用超级管理员上传、且可加载的 detect_model 权重文件。"""
        _ = settings
        rel = resolve_detect_model_rel(str(self.upload_root), self._active_uploaded_rel, check_usable=True)
        if not rel:
            raise ModelUnavailableError(MODEL_UNAVAILABLE_MESSAGE)
        return str((self.upload_root / rel).resolve())

    @staticmethod
    def _get_yolo(weights_path: str):
        return _get_yolo(weights_path)

    def detect_image(
        self,
        image_path,
        settings=None,
        *,
        letterbox_bgr=None,
        letterbox_meta: LetterboxMeta | None = None,
        save_result: bool = True,
        result_stem: str | None = None,
    ):
        """检测图像中的垃圾。

        letterbox_bgr + letterbox_meta：在 640 推理，框与结果图映射/绘制在原图（image_path）上。
        未传 letterbox 时兼容旧逻辑：直接对 image_path 整图推理。
        """
        settings = settings or {}
        try:
            start_time = time.time()
            image_path = str(Path(image_path).resolve()) if image_path else ''

            conf = float(settings.get('confidenceThreshold', DEFAULT_DETECT_CONF))
            conf = max(0.05, min(0.99, conf))
            iou = float(settings.get('nmsThreshold', DEFAULT_DETECT_NMS))
            iou = max(0.1, min(0.95, iou))
            overlay_mode = settings.get("overlayMode") or "all"

            weights_path = self._resolve_weights_path(settings)
            model = self._get_yolo(weights_path)

            if letterbox_bgr is not None and letterbox_meta is not None:
                infer_img = letterbox_bgr
                display_img = None
                if save_result:
                    if not image_path:
                        raise ValueError("缺少原图路径")
                    display_img = cv2.imread(image_path)
                    if display_img is None:
                        raise ValueError("无法读取原始图像文件")
            else:
                if not image_path:
                    raise ValueError("缺少图像路径")
                display_img = cv2.imread(image_path)
                if display_img is None:
                    raise ValueError("无法读取图像文件")
                infer_img = display_img

            results = model.predict(
                infer_img,
                conf=conf,
                iou=iou,
                verbose=False,
                device='cpu',
                imgsz=640,
            )

            detected_objects = []
            confidence_scores = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        if class_id < 0:
                            continue

                        detected_objects.append({
                            "class": self.class_names[class_id] if class_id < len(self.class_names) else "unknown",
                            "confidence": float(confidence),
                            "bbox": [float(x1), float(y1), float(x2), float(y2)]
                        })
                        confidence_scores.append(float(confidence))

            if letterbox_meta is not None:
                detected_objects = map_detections_to_orig(detected_objects, letterbox_meta)

            result_path = None
            if save_result:
                result_image = self.draw_detections(display_img, detected_objects, overlay_mode)
                result_dir = self.upload_root / 'detects_result'
                result_dir.mkdir(parents=True, exist_ok=True)
                stem = result_stem or Path(os.path.basename(image_path)).stem
                result_filename = f'{stem}_result.jpg'
                result_path = result_dir / result_filename
                if not cv2.imwrite(
                    str(result_path),
                    result_image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                ):
                    raise RuntimeError("结果图像保存失败")
                result_path = str(result_path.relative_to(self.upload_root)).replace('\\', '/')

            processing_time = time.time() - start_time
            model_version = str(settings.get('modelVersion') or '').strip()
            if model_version:
                model_display_name = model_version
            else:
                parts = Path(weights_path).parts
                model_display_name = (
                    parts[-2] if 'detect_model' in parts and len(parts) >= 2 else Path(weights_path).parent.name
                )

            return {
                "detected_objects": detected_objects,
                "confidence_scores": confidence_scores,
                "result_path": result_path,
                "processing_time": processing_time,
                "total_objects": len(detected_objects),
                "model_display_name": model_display_name,
                "model_version": model_version,
            }

        except ModelUnavailableError:
            raise
        except Exception as e:
            raise Exception(f"检测过程中发生错误: {str(e)}")

    def draw_detections(self, image, detections, overlay_mode="all"):
        """在图像上绘制检测结果。线宽与字号随图像边长缩放，避免大图里框/字过小、缩放后发糊。"""
        result_image = image.copy()
        box_only = overlay_mode == "boxOnly"
        h, w = result_image.shape[:2]
        ref = max(float(h), float(w), 1.0)
        line_w = max(2, min(10, int(round(ref * 0.0045))))
        font_scale = max(0.65, min(2.4, ref * 0.00135))
        text_th = max(1, min(3, int(round(font_scale * 1.2))))
        pad_y = max(8, int(round(10 * font_scale / 0.65)))
        pad_x = max(6, int(round(8 * font_scale / 0.65)))

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            class_name = detection["class"]
            confidence = detection["confidence"]
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(result_image, (ix1, iy1), (ix2, iy2), (0, 255, 0), line_w, lineType=cv2.LINE_AA)

            if box_only:
                continue

            label = f"{class_name}: {confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_th
            )
            ty0 = iy1 - th - baseline - pad_y
            if ty0 < 0:
                ty0 = iy2 + pad_y // 2
            tx0 = ix1
            if tx0 + tw + pad_x > w:
                tx0 = max(0, w - tw - pad_x)
            bg_x2 = min(w, tx0 + tw + pad_x)
            bg_y2 = min(h, ty0 + th + baseline + pad_y // 2)
            cv2.rectangle(
                result_image,
                (tx0, ty0),
                (bg_x2, bg_y2),
                (0, 255, 0),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            tx = tx0 + pad_x // 2
            ty = min(h - 1, ty0 + th + baseline)
            cv2.putText(
                result_image,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (15, 15, 15),
                text_th,
                lineType=cv2.LINE_AA,
            )

        return result_image


def clear_yolo_cache(paths=None):
    """训练完成后替换权重时清空缓存；paths 为若干绝对路径字符串时仅移除这些键。"""
    if not paths:
        _yolo_cache.clear()
        return
    for p in paths:
        _yolo_cache.pop(str(Path(p).resolve()), None)


# --- site detect thresholds ---

DETECT_SETTINGS_FILENAME = 'detect_settings.json'
DEFAULT_DETECT_CONF = 0.5
DEFAULT_DETECT_NMS = 0.45
DEFAULT_AUTO_INGEST = 0.85


def _clamp_conf(val: float) -> float:
    return max(0.05, min(0.99, float(val)))


def _clamp_nms(val: float) -> float:
    return max(0.1, min(0.95, float(val)))


def _clamp_auto_ingest(val: float) -> float:
    return max(0.01, min(0.99, float(val)))


def read_detect_settings(upload_folder: str, *, auto_ingest_default: float = DEFAULT_AUTO_INGEST) -> dict[str, Any]:
    """全站检测阈值（uploads/detect_settings.json：conf、nms、autoIngestThreshold）。"""
    path = Path(upload_folder).resolve() / DETECT_SETTINGS_FILENAME
    conf = DEFAULT_DETECT_CONF
    nms = DEFAULT_DETECT_NMS
    auto_ingest = _clamp_auto_ingest(auto_ingest_default)
    update_time = ''
    if path.is_file():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                raw_conf = data.get('conf', data.get('confidenceThreshold'))
                if raw_conf is not None:
                    conf = _clamp_conf(raw_conf)
                raw_nms = data.get('nms', data.get('nmsThreshold'))
                if raw_nms is not None:
                    nms = _clamp_nms(raw_nms)
                raw_auto = data.get('autoIngestThreshold', data.get('auto_ingest_threshold'))
                if raw_auto is not None:
                    auto_ingest = _clamp_auto_ingest(raw_auto)
                update_time = str(data.get('update_time') or '').strip()
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return {
        'conf': conf,
        'nms': nms,
        'autoIngestThreshold': auto_ingest,
        'update_time': update_time,
    }


def write_detect_settings(upload_folder: str, patch: dict) -> dict[str, Any]:
    current = read_detect_settings(upload_folder)
    if 'conf' in patch:
        try:
            current['conf'] = _clamp_conf(patch['conf'])
        except (TypeError, ValueError):
            pass
    elif 'confidenceThreshold' in patch:
        try:
            current['conf'] = _clamp_conf(patch['confidenceThreshold'])
        except (TypeError, ValueError):
            pass
    if 'nms' in patch:
        try:
            current['nms'] = _clamp_nms(patch['nms'])
        except (TypeError, ValueError):
            pass
    elif 'nmsThreshold' in patch:
        try:
            current['nms'] = _clamp_nms(patch['nmsThreshold'])
        except (TypeError, ValueError):
            pass
    if 'autoIngestThreshold' in patch:
        try:
            current['autoIngestThreshold'] = _clamp_auto_ingest(patch['autoIngestThreshold'])
        except (TypeError, ValueError):
            pass
    elif 'auto_ingest_threshold' in patch:
        try:
            current['autoIngestThreshold'] = _clamp_auto_ingest(patch['auto_ingest_threshold'])
        except (TypeError, ValueError):
            pass
    current['update_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    path = Path(upload_folder).resolve() / DETECT_SETTINGS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'conf': current['conf'],
                'nms': current['nms'],
                'autoIngestThreshold': current['autoIngestThreshold'],
                'update_time': current['update_time'],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return current


def apply_site_detect_thresholds(settings: dict | None, upload_folder: str, *, auto_ingest_default: float = DEFAULT_AUTO_INGEST) -> dict:
    """识别请求合并全站 conf / nms / autoIngestThreshold。"""
    site = read_detect_settings(upload_folder, auto_ingest_default=auto_ingest_default)
    out = dict(settings or {})
    out['confidenceThreshold'] = site['conf']
    out['nmsThreshold'] = site['nms']
    out['autoIngestThreshold'] = site['autoIngestThreshold']
    return out


# --- routing ---


def auto_ingest_threshold_from_settings(settings: dict | None, default: float = 0.85) -> float:
    """从 settings.autoIngestThreshold 读取；无则回退 config 默认。"""
    if not isinstance(settings, dict):
        return max(0.01, min(0.99, float(default)))
    raw = settings.get('autoIngestThreshold')
    if raw is None:
        raw = settings.get('auto_ingest_threshold')
    try:
        x = float(raw if raw is not None else default)
    except (TypeError, ValueError):
        x = float(default)
    return max(0.01, min(0.99, x))


def route_detection_sample(flask_app, detection_row, result_dict):
    """
    detection_row: 已 flush 的 DetectionResult（有 id）
    result_dict: detect_image 返回值
    """
    dataset_root = Path(flask_app.config.get('DATASET_ROOT') or flask_app.config['PROJECT_ROOT'] / 'datasets')

    objs = result_dict.get('detected_objects') or []
    per_box = []
    auto_ok = 0
    pending_cnt = 0

    raw_cs = detection_row.confidence_scores
    try:
        cs_obj = json.loads(raw_cs) if raw_cs else {}
    except Exception:
        cs_obj = {}

    upload_root = flask_app.config['UPLOAD_FOLDER']
    cfg_default = float(
        flask_app.config.get('AUTO_INGEST_CONF_THRESHOLD', DEFAULT_AUTO_INGEST)
    )
    site = read_detect_settings(upload_root, auto_ingest_default=cfg_default)
    threshold = auto_ingest_threshold_from_settings(
        cs_obj.get('settings'),
        float(site.get('autoIngestThreshold') or cfg_default),
    )

    if not objs:
        routing = {
            'per_box': [],
            'auto_ingested': 0,
            'pending_count': 0,
            'reason': 'no_detection',
        }
        cs_obj['routing'] = routing
        detection_row.confidence_scores = json.dumps(cs_obj, ensure_ascii=False)
        db.session.commit()
        return routing

    uid = int(detection_row.user_id)
    urow = User.query.get(uid) if uid else None
    uname = getattr(urow, 'username', None) if urow else None

    PendingSample.query.filter_by(
        detection_id=detection_row.id,
        status='pending',
    ).delete(synchronize_session=False)
    db.session.flush()

    for idx, det in enumerate(objs):
        conf = float(det.get('confidence') or 0)
        cls_name = str(det.get('class') or 'unknown')
        bbox = det.get('bbox')
        entry = {
            'index': idx,
            'class': cls_name,
            'confidence': conf,
            'mode': None,
            'dataset_image': None,
            'dataset_label': None,
            'pending_sample_id': None,
            'reason': None,
        }

        if conf >= threshold:
            entry['mode'] = 'auto_approved'
            entry['approved_class'] = cls_name
            entry['approved_bbox'] = bbox
            auto_ok += 1
        else:
            ps = PendingSample(
                detection_id=detection_row.id,
                box_index=idx,
                predicted_class=cls_name,
                confidence=conf,
                bbox_json=json.dumps(bbox, ensure_ascii=False) if bbox is not None else None,
                status='pending',
            )
            db.session.add(ps)
            db.session.flush()
            entry['mode'] = 'pending_review'
            entry['pending_sample_id'] = ps.id
            pending_cnt += 1

        per_box.append(entry)

    routing = {
        'per_box': per_box,
        'auto_ingested': auto_ok,
        'pending_count': pending_cnt,
        'auto_ingest_threshold': threshold,
    }

    cs_obj['routing'] = routing
    cs_obj.setdefault('scores', result_dict.get('confidence_scores') or [])
    detection_row.confidence_scores = json.dumps(cs_obj, ensure_ascii=False)
    db.session.commit()

    if pending_cnt == 0 and auto_ok > 0:
        fin = try_finalize_detection_dataset(
            dataset_root,
            detection_row,
            detection_row.file_path,
            username=uname,
            upload_folder=flask_app.config['UPLOAD_FOLDER'],
        )
        if fin.get('finalized'):
            set_detection_dataset_paths(
                detection_row,
                fin.get('dataset_image'),
                fin.get('dataset_label'),
            )
            db.session.commit()
            routing['dataset_image'] = fin.get('dataset_image')
            routing['dataset_label'] = fin.get('dataset_label')
            routing['dataset_finalized'] = True
            routing['finalize_message'] = fin.get('message')

    return routing
