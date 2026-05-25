from pathlib import Path
import argparse

from ultralytics import YOLO
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent


def _resolve_under_training(p: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (ROOT / p).resolve()


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 垃圾识别训练脚本")
    parser.add_argument(
        "--data",
        default=str(ROOT / "trash4.yaml"),
        help="数据集配置文件（相对 training/ 或绝对路径）",
    )
    parser.add_argument(
        "--weights",
        default="",
        help="预训练权重（相对 training/ 或绝对路径；默认 training/bases/ 下首个 .pt）",
    )
    parser.add_argument("--epochs", type=int, default=120, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="图片尺寸")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patience", type=int, default=20, help="早停")
    parser.add_argument(
        "--project",
        default="runs",
        help="YOLO runs 父目录（相对 training/ 或绝对路径）",
    )
    parser.add_argument(
        "--name",
        default="",
        help="运行目录名；留空时需配合 --job-id 生成（如 best_12）",
    )
    parser.add_argument(
        "--job-id",
        type=int,
        default=0,
        help="训练任务数据库主键 id（与 --name 二选一）",
    )

    args = parser.parse_args()
    data_path = _resolve_under_training(Path(args.data))

    weights_arg = str(args.weights or "").strip()
    if weights_arg:
        weights_path = _resolve_under_training(Path(weights_arg))
    else:
        bases = ROOT / "bases"
        candidates = sorted(bases.glob("*.pt")) + sorted(bases.glob("*.pth"))
        if not candidates:
            parser.error("请通过 --weights 指定权重，或先将 .pt/.pth 放入 training/bases/")
        weights_path = candidates[0].resolve()

    project_dir = _resolve_under_training(Path(args.project))

    run_name = str(args.name or "").strip()
    if not run_name:
        job_id = int(args.job_id or 0)
        if job_id < 1:
            parser.error("未指定 --name 时请提供 --job-id（training_job 表主键）")
        stem = secure_filename(weights_path.stem or 'weights') or 'weights'
        run_name = f'{stem}_{job_id}'

    model = YOLO(str(weights_path))

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        project=str(project_dir),
        name=run_name,
        rect=False,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.1,
        conf=0.5,
        iou=0.45,
        save=True,
        val=True,
        plots=True,
        cache="ram",
        verbose=True,
        workers=0,
    )


if __name__ == "__main__":
    main()
