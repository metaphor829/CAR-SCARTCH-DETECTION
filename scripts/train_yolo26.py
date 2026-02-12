"""YOLO26 训练启动脚本。

仅依赖 Python 3.10 标准库，通过 subprocess 调用外部 YOLO CLI。
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run_prepare_script(python_exec: str, dataset_dir: Path, output: Path, classes_file: Path) -> None:
    """在训练前先校验数据并生成 dataset.yaml。"""
    cmd = [
        python_exec,
        "scripts/prepare_dataset.py",
        "--dataset-dir",
        str(dataset_dir),
        "--output",
        str(output),
        "--classes-file",
        str(classes_file),
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError("数据检查失败，请先修复后再训练")


def build_train_command(args: argparse.Namespace) -> list[str]:
    """组装 YOLO 训练命令参数。"""
    data_yaml = str(args.data)
    command = [
        args.yolo_cmd,
        "train",
        f"model={args.model}",
        f"data={data_yaml}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"mosaic={args.mosaic}",
        f"mixup={args.mixup}",
        f"hsv_v={args.hsv_v}",
        f"hsv_s={args.hsv_s}",
        f"close_mosaic={args.close_mosaic}",
        f"warmup_epochs={args.warmup_epochs}",
        f"lr0={args.lr0}",
        f"weight_decay={args.weight_decay}",
        f"box={args.box}",
        f"cls={args.cls}",
        f"project={args.project}",
        f"name={args.name}",
        f"device={args.device}",
    ]

    if args.workers is not None:
        command.append(f"workers={args.workers}")

    if args.patience is not None:
        command.append(f"patience={args.patience}")

    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="启动 YOLO26 训练")
    parser.add_argument("--python", default=sys.executable, help="Python 解释器路径")
    parser.add_argument("--yolo-cmd", default="yolo", help="YOLO 命令名，默认: yolo")
    parser.add_argument("--model", default="best.pt", help="模型权重路径，默认: best.pt")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"), help="数据集根目录")
    parser.add_argument("--classes-file", type=Path, default=Path("configs/classes.txt"), help="类别文件路径")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="dataset.yaml 路径（若不存在会自动生成）",
    )
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=1280, help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="batch 大小")
    parser.add_argument("--mosaic", type=float, default=1.0, help="mosaic 数据增强概率")
    parser.add_argument("--mixup", type=float, default=0.2, help="mixup 数据增强概率")
    parser.add_argument("--hsv-v", type=float, default=0.6, help="HSV 亮度增强幅度")
    parser.add_argument("--hsv-s", type=float, default=0.8, help="HSV 饱和度增强幅度")
    parser.add_argument("--close-mosaic", type=int, default=20, help="训练末期关闭 mosaic 的轮数")
    parser.add_argument("--warmup-epochs", type=float, default=5.0, help="warmup 轮数")
    parser.add_argument("--lr0", type=float, default=0.001, help="初始学习率")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="权重衰减")
    parser.add_argument("--box", type=float, default=5.0, help="box loss 权重")
    parser.add_argument("--cls", type=float, default=1.5, help="cls loss 权重")
    parser.add_argument("--device", default="0", help="设备，如 0 / cpu")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--patience", type=int, default=50, help="早停 patience")
    parser.add_argument("--project", default="runs/train", help="训练输出目录")
    parser.add_argument("--name", default="car-scratch-yolo26", help="本次训练名称")
    args = parser.parse_args()

    try:
        # 关键步骤：训练前先执行数据校验，确保目录结构和标签匹配关系正确。
        run_prepare_script(args.python, args.dataset_dir, args.data, args.classes_file)

        if not args.data.exists():
            raise FileNotFoundError(f"找不到数据配置文件: {args.data}")

        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型权重文件: {model_path}")

        command = build_train_command(args)
        print("执行命令:")
        print(" ".join(shlex.quote(part) for part in command))

        # 关键步骤：将训练进程输出直接透传到终端，便于实时观察 loss、mAP 等指标。
        result = subprocess.run(command, check=False)
        return result.returncode
    except FileNotFoundError as exc:
        print(f"命令或文件不存在: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
