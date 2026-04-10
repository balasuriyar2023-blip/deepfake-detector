import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from facenet_pytorch import MTCNN
import matplotlib.cm as cm
import pandas as pd
import time

# ── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Constants ─────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Real', 'Fake']
MODEL_PATH = 'model_p3.pth'

# ── Load Model ────────────────────────────────────────
@st.cache_resource
def load_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(model.classifier[1].in_features, 2)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ── Load Face Detector ────────────────────────────────
@st.cache_resource
def load_mtcnn():
    return MTCNN(keep_all=False, device=DEVICE)

# ── Transform ─────────────────────────────────────────
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── FIXED Face Detection (clean crop, no weird colors) ─
def detect_face(img, mtcnn):
    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        x1, y1, x2, y2 = map(int, boxes[0])

        # Add padding
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img.width, x2 + pad)
        y2 = min(img.height, y2 + pad)

        face = img.crop((x1, y1, x2, y2))
        return face

    return img  # fallback

# ── Grad-CAM ─────────────────────────────────────────
def get_gradcam(model, img_tensor):
    features = None
    grads = None

    def fwd_hook(module, input, output):
        nonlocal features
        features = output.detach()

    def bwd_hook(module, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0].detach()

    target_layer = model.features[-1][0]

    handle_f = target_layer.register_forward_hook(fwd_hook)
    handle_b = target_layer.register_full_backward_hook(bwd_hook)

    img_tensor = img_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)

    logits = model(img_tensor)
    pred = logits.argmax(1).item()
    prob = torch.softmax(logits, dim=1)[0, pred].item()

    model.zero_grad()
    logits[0, pred].backward()

    handle_f.remove()
    handle_b.remove()

    weights = grads.mean(dim=[2, 3], keepdim=True)
    cam = (weights * features).sum(dim=1).squeeze()

    cam = torch.relu(cam).cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    return cam, pred, prob

def overlay_cam(img_pil, cam):
    img_np = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0
    cam_resized = np.array(Image.fromarray(cam).resize((224, 224)))
    heatmap = cm.jet(cam_resized)[:, :, :3]

    overlay = 0.5 * img_np + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)

    return (overlay * 255).astype(np.uint8)

# ── Explanation ───────────────────────────────────────
def get_reason(prob, pred):
    if pred == 0:
        return "Face appears natural with consistent features."

    if prob > 0.9:
        return "Strong AI-generation signs: texture artifacts, unnatural blending."
    elif prob > 0.7:
        return "Moderate manipulation indicators detected."
    else:
        return "Weak signals — verify with additional tools."

# ── UI ────────────────────────────────────────────────
st.title("🔍 DeepFake Face Detector")
st.markdown("Upload face images to detect whether they are **Real** or **AI-Generated**.")
st.markdown("**Model: EfficientNet-B0 | Accuracy: 91.69% | AUC: 0.9764**")
st.divider()

uploaded_files = st.file_uploader(
    "Upload Image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    try:
        model = load_model()
        mtcnn = load_mtcnn()
    except FileNotFoundError:
        st.error("Model file not found. Please upload model_p3.pth.")
        st.stop()

    results = []

    for uploaded in uploaded_files:
        img = Image.open(uploaded).convert("RGB")

        start = time.time()

        face = detect_face(img, mtcnn)
        img_tensor = eval_tf(face)

        cam, pred, prob = get_gradcam(model, img_tensor)
        cam_img = overlay_cam(face, cam)

        end = time.time()

        label = CLASS_NAMES[pred]

        # UI Output
        st.subheader(f"{'🟢' if pred == 0 else '🔴'} {uploaded.name}")

        if pred == 1:
            st.warning("Likely AI-generated")
        else:
            st.success("Likely Real")

        st.metric("Confidence", f"{prob*100:.2f}%")
        st.caption(f"Inference time: {end-start:.2f}s")

        st.info(get_reason(prob, pred))

        col1, col2 = st.columns(2)
        with col1:
            st.image(face.resize((224, 224)), caption="Detected Face", use_container_width=True)
        with col2:
            st.image(cam_img, caption="Grad-CAM", use_container_width=True)

        st.divider()

        results.append({
            "Image": uploaded.name,
            "Prediction": label,
            "Confidence": prob
        })

    # Download results
    df = pd.DataFrame(results)

    st.download_button(
        "📥 Download Results",
        df.to_csv(index=False),
        "results.csv",
        "text/csv"
    )
