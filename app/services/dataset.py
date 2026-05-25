"""训练集：routing 元数据、YOLO 序号命名、images/labels 写入。"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import cv2

from app import db
from app.models import PendingSample
from app.utils import resolve_stored_file_path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


# --- routing meta ---


def load_confidence_scores(detection_row) -> dict:
    raw = getattr(detection_row, 'confidence_scores', None)
    try:
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def save_confidence_scores(detection_row, cs_obj: dict) -> None:
    detection_row.confidence_scores = json.dumps(cs_obj, ensure_ascii=False)


def update_per_box_entry(detection_row, box_index: int, **fields) -> None:
    cs = load_confidence_scores(detection_row)
    routing = cs.setdefault('routing', {})
    per_box = routing.setdefault('per_box', [])
    idx = int(box_index)
    target = None
    for entry in per_box:
        if int(entry.get('index', -1)) == idx:
            target = entry
            break
    if target is None:
        target = {'index': idx}
        per_box.append(target)
    target.update(fields)
    routing['per_box'] = per_box
    save_confidence_scores(detection_row, cs)


def set_detection_dataset_paths(detection_row, dataset_image: str | None, dataset_label: str | None) -> None:
    cs = load_confidence_scores(detection_row)
    routing = cs.setdefault('routing', {})
    if dataset_image:
        routing['dataset_image'] = dataset_image
        bn = str(dataset_image).replace('\\', '/').split('/')[-1]
        if bn:
            routing['dataset_stem'] = Path(bn).stem
    if dataset_label:
        routing['dataset_label'] = dataset_label
    routing['dataset_finalized'] = bool(dataset_image and dataset_label)
    save_confidence_scores(detection_row, cs)


# --- dataset naming ---


def normalize_image_ext(path: Path | str) -> str:
    p = Path(path)
    ext = p.suffix.lower() if p.suffix else '.jpg'
    if ext not in IMAGE_EXTS:
        ext = '.jpg'
    if ext == '.jpeg':
        ext = '.jpg'
    return ext


def max_dataset_sequence_number(images_dir: Path) -> int:
    """images/ 下纯数字文件名（如 1.jpg）的最大序号。"""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        return 0
    max_n = 0
    for p in images_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            n = int(p.stem)
        except ValueError:
            continue
        if n > max_n:
            max_n = n
    return max_n


def get_routing_dataset_stem(detection_row) -> str | None:
    raw = getattr(detection_row, 'confidence_scores', None)
    try:
        cs = json.loads(raw) if raw else {}
    except Exception:
        cs = {}
    routing = cs.get('routing') or {}
    stem = routing.get('dataset_stem')
    if stem is not None:
        s = str(stem).strip()
        if s.isdigit():
            return s
    img_rel = routing.get('dataset_image') or ''
    if img_rel:
        bn = Path(str(img_rel).replace('\\', '/')).name
        st = Path(bn).stem
        if st.isdigit():
            return st
    return None


def resolve_dataset_basename(
    dataset_root: Path,
    detection_row,
    source_image_path: Path | str,
) -> tuple[str, str]:
    """
    返回 (basename, dataset_stem)。
    已分配过序号的 detection 复用同一 stem；否则在现有最大序号 +1。
    """
    dataset_root = Path(dataset_root).resolve()
    images_dir = dataset_root / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    ext = normalize_image_ext(source_image_path)
    existing_stem = get_routing_dataset_stem(detection_row)
    if existing_stem:
        return f'{existing_stem}{ext}', existing_stem

    next_n = max_dataset_sequence_number(images_dir) + 1
    stem = str(next_n)
    return f'{stem}{ext}', stem


# --- ingest ---


def _read_class_names(dataset_root: Path):
    p = dataset_root / 'labels' / 'classes.txt'
    if not p.is_file():
        return ['recyclable', 'other', 'hazardous', 'kitchen']
    lines = p.read_text(encoding='utf-8', errors='ignore').strip().splitlines()
    return [x.strip() for x in lines if x.strip()]


def class_name_to_id(class_name: str, class_names) -> int:
    n = (class_name or '').strip()
    for i, c in enumerate(class_names):
        if c == n:
            return i
    return 0


def xyxy_to_yolo_line(class_id: int, x1, y1, x2, y2, w_img: float, h_img: float) -> str:
    w_img = max(float(w_img), 1.0)
    h_img = max(float(h_img), 1.0)
    x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
    bw = max(x2 - x1, 0.0)
    bh = max(y2 - y1, 0.0)
    cx = (x1 + x2) / 2.0 / w_img
    cy = (y1 + y2) / 2.0 / h_img
    nw = bw / w_img
    nh = bh / h_img
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 1e-6), 1.0)
    nh = min(max(nh, 1e-6), 1.0)
    return f'{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}'


def bbox_from_detection(det: dict):
    b = det.get('bbox') or []
    if len(b) >= 4:
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    return None


def _parse_detected_objects(detection_row) -> list:
    raw = getattr(detection_row, 'detected_objects', None)
    if not raw:
        return []
    try:
        arr = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []
    return arr if isinstance(arr, list) else []


def _routing_per_box(detection_row) -> list:
    raw = getattr(detection_row, 'confidence_scores', None)
    try:
        cs = json.loads(raw) if raw else {}
    except Exception:
        cs = {}
    return list((cs.get('routing') or {}).get('per_box') or [])


def count_pending_boxes(detection_id: int) -> int:
    return int(
        PendingSample.query.filter_by(detection_id=int(detection_id), status='pending').count()
    )


def collect_ingest_boxes(detection_row) -> list[dict]:
    """
    汇总应写入 YOLO 标注的全部框：高置信自动采纳（routing）+ 已审核入库（pending_sample）。
    返回 [{box_index, class_name, bbox_xyxy}, ...]，按 box_index 排序。
    """
    det_id = int(detection_row.id)
    objs = _parse_detected_objects(detection_row)
    per_box = _routing_per_box(detection_row)
    by_idx: dict[int, dict] = {}

    for ps in PendingSample.query.filter_by(detection_id=det_id, status='ingested').all():
        try:
            bbox = json.loads(ps.bbox_json) if ps.bbox_json else None
        except Exception:
            bbox = None
        if not bbox or len(bbox) < 4:
            idx = int(ps.box_index or 0)
            if idx < len(objs):
                bbox = bbox_from_detection(objs[idx])
        if not bbox:
            continue
        cls = (ps.corrected_class or ps.predicted_class or '').strip()
        if not cls:
            continue
        by_idx[int(ps.box_index)] = {
            'box_index': int(ps.box_index),
            'class_name': cls,
            'bbox_xyxy': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
        }

    for entry in per_box:
        idx = int(entry.get('index', -1))
        if idx < 0 or idx in by_idx:
            continue
        mode = entry.get('mode') or ''
        if mode not in ('auto_approved', 'auto_dataset'):
            continue
        cls = (entry.get('approved_class') or entry.get('class') or '').strip()
        bbox = entry.get('approved_bbox')
        if bbox is None and idx < len(objs):
            bbox = bbox_from_detection(objs[idx])
        if not cls or not bbox:
            continue
        by_idx[idx] = {
            'box_index': idx,
            'class_name': cls,
            'bbox_xyxy': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
        }

    return [by_idx[k] for k in sorted(by_idx.keys())]


def _resolve_ingest_image_path(image_path: str, upload_folder: str | None = None) -> Path | None:
    p = None
    if upload_folder:
        p = resolve_stored_file_path(image_path, upload_folder)
    if not p:
        raw = str(image_path or '').strip()
        if raw and os.path.isabs(raw):
            p = os.path.normpath(raw)
    if not p or not os.path.isfile(p):
        return None
    return Path(p).resolve()


def finalize_detection_dataset(
    dataset_root: Path,
    detection_row,
    image_path: str,
    username: str | None = None,
    upload_folder: str | None = None,
):
    """
    复制原图一次（若尚未存在），写入包含全部框的 labels/{stem}.txt。
    返回 (image_rel, label_rel, error_msg, box_count)。
    """
    _ = username
    dataset_root = Path(dataset_root).resolve()
    img_path = _resolve_ingest_image_path(image_path, upload_folder)
    if img_path is None:
        return None, None, '原图不存在', 0
    boxes = collect_ingest_boxes(detection_row)
    if not boxes:
        return None, None, '没有可入库的检测框', 0

    im = cv2.imread(str(img_path))
    if im is None:
        return None, None, '无法读取图像', 0
    h, w = im.shape[:2]

    class_names = _read_class_names(dataset_root)
    (dataset_root / 'images').mkdir(parents=True, exist_ok=True)
    (dataset_root / 'labels').mkdir(parents=True, exist_ok=True)

    basename, stem = resolve_dataset_basename(dataset_root, detection_row, img_path)
    dst_img = dataset_root / 'images' / basename

    if not dst_img.is_file():
        shutil.copy2(img_path, dst_img)

    lines = []
    for item in boxes:
        cls_id = class_name_to_id(item['class_name'], class_names)
        x1, y1, x2, y2 = item['bbox_xyxy']
        lines.append(xyxy_to_yolo_line(cls_id, x1, y1, x2, y2, w, h))

    dst_lbl = dataset_root / 'labels' / f'{stem}.txt'
    dst_lbl.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    rel_img = f'images/{basename}'.replace('\\', '/')
    rel_lbl = f'labels/{stem}.txt'.replace('\\', '/')
    return rel_img, rel_lbl, None, len(lines)


def try_finalize_detection_dataset(
    dataset_root: Path,
    detection_row,
    image_path: str,
    username: str | None = None,
    upload_folder: str | None = None,
):
    """
    当该 detection 无待审核框时，写入/更新整图标注。
    返回 dict: finalized, pending_remaining, dataset_image, dataset_label, box_count, message
    """
    pending = count_pending_boxes(detection_row.id)
    if pending > 0:
        return {
            'finalized': False,
            'pending_remaining': pending,
            'dataset_image': None,
            'dataset_label': None,
            'box_count': 0,
            'message': f'本图尚有 {pending} 个框待审核，暂未写入数据集',
        }

    rel_img, rel_lbl, err, box_count = finalize_detection_dataset(
        dataset_root,
        detection_row,
        image_path,
        username=username,
        upload_folder=upload_folder,
    )
    if err:
        return {
            'finalized': False,
            'pending_remaining': 0,
            'dataset_image': None,
            'dataset_label': None,
            'box_count': 0,
            'message': err,
        }
    if box_count == 0:
        return {
            'finalized': False,
            'pending_remaining': 0,
            'dataset_image': None,
            'dataset_label': None,
            'box_count': 0,
            'message': '本图无采纳框，未写入数据集',
        }

    return {
        'finalized': True,
        'pending_remaining': 0,
        'dataset_image': rel_img,
        'dataset_label': rel_lbl,
        'box_count': box_count,
        'message': f'已写入训练集：{rel_img} + {rel_lbl}（共 {box_count} 个框）',
    }
