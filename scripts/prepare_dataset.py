"""检查 YOLO 数据集并生成 dataset.yaml。

仅依赖 Python 3.10 标准库。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def iter_image_files(folder: Path) -> Iterable[Path]:
    """遍历指定目录下的图像文件。"""
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def read_classes(classes_file: Path) -> list[str]:
    """读取类别配置（每行一个类别）。"""
    if not classes_file.exists():
        raise FileNotFoundError(f"类别文件不存在: {classes_file}")

    # 使用 utf-8-sig 兼容带 BOM 的文本，避免首个类别名前出现隐藏字符。
    classes = [line.strip() for line in classes_file.read_text(encoding="utf-8-sig").splitlines()]
    classes = [c for c in classes if c]
    if not classes:
        raise ValueError("类别文件为空，请至少配置一个类别")
    return classes


def check_split(dataset_dir: Path, split: str) -> tuple[int, int]:
    """检查单个数据分片的图像/标签是否匹配，返回 (图像数, 缺失标签数)。"""
    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split

    if not image_dir.exists():
        raise FileNotFoundError(f"缺少目录: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"缺少目录: {label_dir}")

    image_files = list(iter_image_files(image_dir))
    missing_labels = 0

    # 关键步骤：按照同名规则检查 image 与 label 是否一一对应（xxx.jpg -> xxx.txt）。
    for image_path in image_files:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1

    return len(image_files), missing_labels


def build_dataset_yaml_text(dataset_dir: Path, classes: list[str]) -> str:
    """构建 YOLO dataset.yaml 文本。"""
    lines = [
        f"path: {dataset_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(classes)}",
        "names:",
    ]

    for idx, class_name in enumerate(classes):
        lines.append(f"  {idx}: {class_name}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 YOLO26 数据集并生成 dataset.yaml")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="数据集根目录，默认: dataset",
    )
    parser.add_argument(
        "--classes-file",
        type=Path,
        default=Path("configs/classes.txt"),
        help="类别文件路径，默认: configs/classes.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="输出 YAML 路径，默认: configs/dataset.yaml",
    )

    args = parser.parse_args()

    try:
        classes = read_classes(args.classes_file)

        has_error = False
        for split in SPLITS:
            image_count, missing_labels = check_split(args.dataset_dir, split)
            print(f"[{split}] images={image_count}, missing_labels={missing_labels}")

            # 关键步骤：若存在缺失标签，直接标记错误，避免后续训练中断或学到噪声。
            if missing_labels > 0:
                has_error = True

        yaml_text = build_dataset_yaml_text(args.dataset_dir.resolve(), classes)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml_text, encoding="utf-8")
        print(f"已生成: {args.output}")

        if has_error:
            print("检测到缺失标签文件，请修复后再训练。", file=sys.stderr)
            return 2

        return 0
    except Exception as exc:  # pragma: no cover
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
