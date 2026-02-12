import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

APP_TITLE = "YOLO 自動車損傷検出システム"
SIDEBAR_TITLE = "モデル設定"
TASK_LABEL = "タスク選択"
MODEL_LABEL = "モデル選択"
CONF_LABEL = "信頼度しきい値"
UPLOAD_LABEL = "検出画像をアップロード"
RUN_LABEL = "認識開始"


def find_pt_models(root: Path) -> list[Path]:
    """ルート直下の .pt モデルを探索する。"""
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".pt"])


@st.cache_resource(show_spinner=False)
def load_model(model_path: Path) -> YOLO:
    """YOLO モデルを読み込む（キャッシュ）。"""
    return YOLO(str(model_path))


def image_bytes_to_bgr(uploaded_bytes: bytes) -> np.ndarray:
    """アップロードされた画像バイト列を BGR 画像に変換する。"""
    pil_img = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb_image(bgr: np.ndarray) -> Image.Image:
    """BGR 配列を RGB PIL 画像に変換する。"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def render_results(results) -> tuple[Image.Image, list[tuple[str, float]]]:
    """推論結果から描画済み画像と (クラス名, 信頼度) を作成する。"""
    plotted_bgr = results[0].plot()
    plotted_img = bgr_to_rgb_image(plotted_bgr)

    detections: list[tuple[str, float]] = []
    boxes = results[0].boxes
    if boxes is not None and boxes.conf is not None:
        confs = boxes.conf.tolist()
        clss = boxes.cls.tolist() if boxes.cls is not None else []
        names = results[0].names
        for cls_id, conf in zip(clss, confs):
            class_name = names.get(int(cls_id), str(int(cls_id)))
            detections.append((class_name, float(conf)))

    return plotted_img, detections


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(APP_TITLE)

    with st.sidebar:
        st.header(SIDEBAR_TITLE)
        st.selectbox(TASK_LABEL, ["Detection"], index=0)

        root = Path.cwd()
        model_paths = find_pt_models(root)
        if not model_paths:
            st.warning("ルートディレクトリに .pt モデルが見つかりません。")
        model_labels = [p.name for p in model_paths] if model_paths else ["(モデルなし)"]
        selected_label = st.selectbox(MODEL_LABEL, model_labels, index=0)

        conf = st.slider(CONF_LABEL, min_value=0.0, max_value=1.0, value=0.25, step=0.01)

        uploaded = st.file_uploader(UPLOAD_LABEL, type=["jpg", "jpeg", "png"])

        run_clicked = st.button(RUN_LABEL, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.subheader("検出前")
    col2.subheader("検出後")

    if run_clicked:
        if uploaded is None:
            st.error("画像がアップロードされていません。")
            return
        if not model_paths:
            st.error("モデルが見つからないため実行できません。")
            return

        model_path = root / selected_label
        try:
            model = load_model(model_path)
        except Exception as exc:
            st.error(f"モデルの読み込みに失敗しました: {exc}")
            return

        with st.spinner("推論中..."):
            bgr = image_bytes_to_bgr(uploaded.getvalue())
            results = model.predict(source=bgr, conf=conf)

        orig_img = bgr_to_rgb_image(bgr)
        plotted_img, detections = render_results(results)

        col1.image(orig_img, caption="入力画像", use_container_width=True)
        col2.image(plotted_img, caption="推論結果", use_container_width=True)

        if detections:
            st.success(f"検出数: {len(detections)}")
            st.table(
                {
                    "クラス": [d[0] for d in detections],
                    "信頼度": [f"{d[1]:.3f}" for d in detections],
                }
            )
        else:
            st.info("検出結果はありません。")


if __name__ == "__main__":
    main()
