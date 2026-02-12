# Car Scratch Detection (YOLO26)

该项目用于训练车辆划痕自动检测模型，按 YOLO 常见数据格式组织。

## 目录结构

```text
Car-Scratch-Detection/
├─ dataset/
│  ├─ images/
│  │  ├─ train/
│  │  ├─ val/
│  │  └─ test/
│  └─ labels/
│     ├─ train/
│     ├─ val/
│     └─ test/
├─ configs/
│  ├─ classes.txt
│  └─ dataset.yaml            # 自动生成
├─ scripts/
│  ├─ prepare_dataset.py
│  └─ train_yolo26.py
├─ app.py
└─ requirements.txt
```

## 依赖安装

```powershell
python -m pip install -r requirements.txt
```

如需 CUDA 版 PyTorch，请根据你的 CUDA 版本安装对应的官方轮子（建议参考 PyTorch 官网）。

## 使用说明

1. 将标注好的数据放入 `dataset/` 对应目录。
2. 根据类别修改 `configs/classes.txt`（每行一个类别名）。
3. 运行数据检查与配置生成：

```powershell
python scripts/prepare_dataset.py
```

4. 启动训练（默认使用根目录下的 `best.pt` 作为初始权重）：

```powershell
python scripts/train_yolo26.py
```

如需指定模型：

```powershell
python scripts/train_yolo26.py --model best.pt
```

## Streamlit 前端

```powershell
streamlit run app.py
```

## 说明

- 脚本仅使用 Python 3.10 标准库。
- 训练本体依赖你本机已安装的 YOLO 命令行工具（脚本通过 `subprocess` 调用）。
- 如果你的命令不是 `yolo`，可用 `--yolo-cmd` 指定。
