import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Constants ──────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Real', 'Fake']
MODEL_PATH = 'model_p3.pth'   # ← update this to your model path

# ── Load model (cached) ────────────────────────────────
@st.cache_resource
def load_model():
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(m.classifier[1].in_features, 2)
    )
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.to(DEVICE)
    m.eval()
    return m

# ── Transform ──────────────────────────────────────────
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Grad-CAM (manual, no extra lib needed) ─────────────
def get_gradcam(model, img_tensor):
    features = None
    grads    = None

    def fwd_hook(module, input, output):
        nonlocal features
        features = output.detach()

    def bwd_hook(module, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0].detach()

    handle_f = model.features[-1].register_forward_hook(fwd_hook)
    handle_b = model.features[-1].register_full_backward_hook(bwd_hook)

    img_tensor = img_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)
    logits = model(img_tensor)
    pred   = logits.argmax(1).item()
    prob   = torch.softmax(logits, dim=1)[0, pred].item()

    model.zero_grad()
    logits[0, pred].backward()

    handle_f.remove()
    handle_b.remove()

    weights   = grads.mean(dim=[2, 3], keepdim=True)
    cam       = (weights * features).sum(dim=1).squeeze()
    cam       = torch.relu(cam).cpu().numpy()
    cam       = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    return cam, pred, prob

def overlay_cam(img_pil, cam):
    img_np = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0
    cam_resized = np.array(Image.fromarray(cam).resize((224, 224), Image.BILINEAR))
    heatmap = cm.jet(cam_resized)[:, :, :3]
    overlay = 0.5 * img_np + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)

# ── UI ─────────────────────────────────────────────────
st.title("🔍 DeepFake Face Detector")
st.markdown("Upload a face image — the model will tell you if it's **Real** or **AI-Generated (Fake)**.")
st.markdown(f"**Model accuracy: 91.69% | AUC: 0.9764**")
st.divider()

uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    try:
        model = load_model()
    except FileNotFoundError:
        st.error(f"Model file '{MODEL_PATH}' not found. Update MODEL_PATH at the top of app.py to point to your saved model_p3.pth file.")
        st.stop()

    img_tensor = eval_tf(img)

    with st.spinner("Analyzing..."):
        cam, pred, prob = get_gradcam(model, img_tensor)
        cam_img = overlay_cam(img, cam)

    # Results
    label = CLASS_NAMES[pred]
    color = "🟢" if pred == 0 else "🔴"

    st.markdown(f"## {color} Prediction: **{label}**")

    conf_col, _ = st.columns([1, 2])
    with conf_col:
        st.metric("Confidence", f"{prob * 100:.1f}%")

    # Confidence bar
    real_prob = 1 - prob if pred == 1 else prob
    fake_prob = prob if pred == 1 else 1 - prob
    st.markdown("**Class Probabilities**")
    st.progress(real_prob, text=f"Real: {real_prob*100:.1f}%")
    st.progress(fake_prob, text=f"Fake: {fake_prob*100:.1f}%")

    st.divider()

    # Side by side: original + grad-cam
    st.markdown("### 🧠 Grad-CAM — What the model focused on")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img.resize((224, 224)), caption="Original Image", use_container_width=True)
    with col2:
        st.image(cam_img, caption=f"Grad-CAM Heatmap (Pred: {label})", use_container_width=True)

    st.markdown("""
    > **How to read Grad-CAM:** Red/yellow areas = regions the model focused on most.
    > For fake faces, it typically highlights unnatural skin texture, eye asymmetry, and hair boundaries.
    """)

    st.divider()
    st.caption("Model: EfficientNet-B0 | Trained on 140k Real & Fake Faces | 3-Phase Transfer Learning")
