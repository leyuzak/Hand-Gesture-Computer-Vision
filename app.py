# app.py
import io
import os
import pandas as pd
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v2

st.set_page_config(page_title="Hand Gesture Recognition", layout="centered")

MODEL_PATH = "MobileNetV2_best.pt"
META_PATH = "leapgestrecog_metadata.csv"


def load_metadata(meta_path: str):
    if not os.path.exists(meta_path):
        return None, None

    df = pd.read_csv(meta_path)

    name_cols = [c for c in df.columns if c.lower() in ["label", "class", "gesture", "category", "name", "classname"]]
    id_cols = [c for c in df.columns if c.lower() in ["class_id", "id", "label_id", "category_id"]]

    name_col = name_cols[0] if name_cols else None
    id_col = id_cols[0] if id_cols else None

    if name_col is None:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        name_col = obj_cols[0] if obj_cols else df.columns[0]

    if id_col is not None:
        tmp = df[[id_col, name_col]].dropna().drop_duplicates()
        tmp[id_col] = tmp[id_col].astype(int)
        id_to_name = dict(zip(tmp[id_col].tolist(), tmp[name_col].astype(str).tolist()))
        max_id = max(id_to_name.keys())
        class_names = [id_to_name.get(i, f"class_{i}") for i in range(max_id + 1)]
        return class_names, id_to_name

    class_names = sorted(df[name_col].dropna().astype(str).unique().tolist())
    id_to_name = {i: n for i, n in enumerate(class_names)}
    return class_names, id_to_name


def build_model(num_classes: int):
    model = mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def try_load_model(model_path: str, num_classes: int, device: torch.device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")

    # TorchScript dene
    try:
        m = torch.jit.load(model_path, map_location=device)
        m.eval()
        return m
    except Exception:
        pass

    obj = torch.load(model_path, map_location=device)

    # Full model
    if hasattr(obj, "eval") and hasattr(obj, "forward"):
        obj.eval()
        return obj

    # State dict
    if isinstance(obj, dict):
        state = obj["state_dict"] if ("state_dict" in obj and isinstance(obj["state_dict"], dict)) else obj

        model = build_model(num_classes).to(device)
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)
        model.eval()
        return model

    raise RuntimeError("Model formatı anlaşılamadı.")


@st.cache_resource
def get_model_and_labels():
    class_names, _ = load_metadata(META_PATH)
    if class_names is None:
        # metadata yoksa fallback
        class_names = [f"class_{i}" for i in range(10)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = try_load_model(MODEL_PATH, num_classes=len(class_names), device=device)
    return model, class_names, device


def preprocess_pil(img: Image.Image):
    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    if img.mode != "RGB":
        img = img.convert("RGB")
    return tfm(img).unsqueeze(0)


def predict(model, class_names, device, img: Image.Image, topk: int = 5):
    x = preprocess_pil(img).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

    k = min(topk, len(class_names))
    idxs = probs.argsort()[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in idxs]


# ---------------- UI ----------------
st.title("🤟 Hand Gesture Recognition")

# Dosyalar yoksa arayüzde kutu basmadan net hata verelim
if not os.path.exists(MODEL_PATH):
    st.error(f"Model dosyası bulunamadı: {MODEL_PATH}")
    st.stop()
if not os.path.exists(META_PATH):
    st.error(f"Metadata dosyası bulunamadı: {META_PATH}")
    st.stop()

try:
    model, class_names, device = get_model_and_labels()
except Exception as e:
    st.error(f"Model yüklenemedi: {e}")
    st.stop()

st.caption(f"Cihaz: **{device}** | Sınıf sayısı: **{len(class_names)}**")

tab1, tab2 = st.tabs(["📷 Kamera (foto)", "🖼️ Resim yükle"])

with tab1:
    cam = st.camera_input("Kameradan bir fotoğraf çek")
    if cam is not None:
        img = Image.open(io.BytesIO(cam.getvalue()))
        st.image(img, caption="Girdi", use_container_width=True)

        results = predict(model, class_names, device, img, topk=5)
        best_name, best_p = results[0]

        st.subheader("Tahmin")
        st.success(f"**{best_name}** ({best_p*100:.2f}%)")

        st.write("Top-5")
        st.table(
            pd.DataFrame(results, columns=["class", "prob"])
            .assign(prob=lambda d: (d["prob"] * 100).round(2).astype(str) + " %")
        )

with tab2:
    up = st.file_uploader("Bir görüntü yükle (jpg/png/webp)", type=["jpg", "jpeg", "png", "webp"])
    if up is not None:
        img = Image.open(up)
        st.image(img, caption="Girdi", use_container_width=True)

        results = predict(model, class_names, device, img, topk=5)
        best_name, best_p = results[0]

        st.subheader("Tahmin")
        st.success(f"**{best_name}** ({best_p*100:.2f}%)")

        st.write("Top-5")
        st.table(
            pd.DataFrame(results, columns=["class", "prob"])
            .assign(prob=lambda d: (d["prob"] * 100).round(2).astype(str) + " %")
        )

st.divider()
st.caption("İpucu: Canlı video istersen `streamlit-webrtc` ile gerçek zamanlı akış ekleyebiliriz.")
