"""
1) 删除「没有对应图片」的标签文件（labels/classes.txt 不处理）。
2) 将 images 下图片按排序后从 1 起连续编号；同名标签同步改为 1.txt、2.txt…

使用两阶段临时文件名，避免重命名过程中覆盖已有文件。
"""
from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TEMP_PREFIX = "__ds_renum_tmp__"

def stem_sort_key(stem: str) -> tuple:
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem.lower())

def image_stems(images_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not images_dir.is_dir():
        return out
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            if p.stem not in out:
                out[p.stem] = p
    return out

def label_paths(labels_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not labels_dir.is_dir():
        return out
    for p in labels_dir.iterdir():
        if p.suffix.lower() == ".txt" and p.name.lower() != "classes.txt":
            out[p.stem] = p
    return out

def main() -> None:
    parser = argparse.ArgumentParser(description="删除孤儿标签并按 1..N 重命名图与标签")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将执行的操作，不删除、不重命名",
    )
    args = parser.parse_args()
    dry = args.dry_run

    base = Path(__file__).resolve().parent
    images_dir = base / "images"
    labels_dir = base / "labels"

    if not images_dir.is_dir():
        raise SystemExit(f"缺少目录: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"缺少目录: {labels_dir}")

    images = image_stems(images_dir)
    labels = label_paths(labels_dir)

    orphan_labels = sorted(
        ((s, labels[s]) for s in labels if s not in images),
        key=lambda x: stem_sort_key(x[0]),
    )

    if orphan_labels:
        print(f"将删除无对应图片的标签 {len(orphan_labels)} 个:")
        for s, p in orphan_labels[:30]:
            print(f"  {p.name}")
        if len(orphan_labels) > 30:
            print(f"  ... 共 {len(orphan_labels)} 个")
        if not dry:
            for _, p in orphan_labels:
                p.unlink(missing_ok=True)
    else:
        print("无孤儿标签需删除。")

    labels = label_paths(labels_dir)
    sorted_stems = sorted(images.keys(), key=stem_sort_key)
    rows: list[tuple[Path, bool, str]] = []
    for stem in sorted_stems:
        img_p = images[stem]
        has_lbl = stem in labels
        rows.append((img_p, has_lbl, img_p.suffix))

    print(f"\n将重命名图片 {len(rows)} 张（及同名标签）为 1..{len(rows)}")
    if dry and rows:
        for i, (img_p, has_lbl, suf) in enumerate(rows[:10], start=1):
            extra = f" + {img_p.stem}.txt" if has_lbl else " (无标签)"
            print(f"  {i}{suf} <- {img_p.name}{extra}")
        if len(rows) > 10:
            print(f"  ... 共 {len(rows)} 条")
    if dry:
        print("\n[--dry-run] 未执行删除与重命名。")
        return

    for i, (img_p, has_lbl, suf) in enumerate(rows, start=1):
        tmp_img = images_dir / f"{TEMP_PREFIX}{i:06d}{suf}"
        img_p.rename(tmp_img)
        if has_lbl:
            lbl_p = labels_dir / f"{img_p.stem}.txt"
            tmp_lbl = labels_dir / f"{TEMP_PREFIX}{i:06d}.txt"
            lbl_p.rename(tmp_lbl)

    for i, (_, has_lbl, suf) in enumerate(rows, start=1):
        tmp_img = images_dir / f"{TEMP_PREFIX}{i:06d}{suf}"
        final_img = images_dir / f"{i}{suf}"
        tmp_img.rename(final_img)
        if has_lbl:
            tmp_lbl = labels_dir / f"{TEMP_PREFIX}{i:06d}.txt"
            final_lbl = labels_dir / f"{i}.txt"
            tmp_lbl.rename(final_lbl)

    print("完成。labels/classes.txt 未改动。")
    print("若 train.txt / val.txt 等仍写旧路径，请重新生成列表。")

if __name__ == "__main__":
    main()
