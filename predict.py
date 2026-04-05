"""
predict.py — DeepFake Face Detector Inference Script
------------------------------------------------------
Usage:
    python predict.py --image path/to/face.jpg
    python predict.py --image path/to/face.jpg --model model_p3.pth
    python predict.py --image path/to/face.jpg --model model_p3.pth --gradcam

Output:
    Prediction : Fake
    Confidence : 94.3%
    Real prob  : 5.7%
    Fake prob  : 94.3%
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ── Constants ──────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
CLASS_NAMES = ['Real', 'Fake']

# ── Model builder ──────────────────────────────────────
def build_model(model_path, device):
    m = efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(m.classifier[1].in_features, 2)
    )
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.to(device)
    m.eval()
    return m

# ── Transform ──────────────────────────────────────────
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Grad-CAM ───────────────────────────────────────────
def get_gradcam(model, img_tensor, device, pred_class):
    features_out = {}
    grads_out    = {}

    def fwd(module, inp, out):
        features_out['f'] = out.detach()

    def bwd(module, gin, gout):
        grads_out['g'] = gout[0].detach()

    hf = model.features[-1].register_forward_hook(fwd)
    hb = model.features[-1].register_full_backward_hook(bwd)

    t = img_tensor.unsqueeze(0).to(device).requires_grad_(True)
    logits = model(t)
    model.zero_grad()
    logits[0, pred_class].backward()

    hf.remove()
    hb.remove()

    w   = grads_out['g'].mean(dim=[2, 3], keepdim=True)
    cam = (w * features_out['f']).sum(dim=1).squeeze()
    cam = torch.relu(cam).cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def save_gradcam(img_pil, cam, output_path, label):
    import matplotlib.pyplot as plt
    import matplotlib.cm as colormap

    img_np      = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0
    cam_resized = np.array(Image.fromarray(cam).resize((224, 224), Image.BILINEAR))
    heatmap     = colormap.jet(cam_resized)[:, :, :3]
    overlay     = np.clip(0.5 * img_np + 0.5 * heatmap, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM  |  Pred: {label}")
    axes[1].axis('off')
    plt.suptitle("DeepFake Detector — Explainability", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Grad-CAM saved → {output_path}")

# ── Main ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DeepFake Face Detector")
    parser.add_argument('--image',   required=True,             help='Path to input image')
    parser.add_argument('--model',   default='model_p3.pth',   help='Path to model weights')
    parser.add_argument('--gradcam', action='store_true',       help='Save Grad-CAM visualization')
    args = parser.parse_args()

    # Validate image
    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load
    print(f"Loading model from {args.model} ...")
    model = build_model(args.model, device)

    img        = Image.open(args.image).convert('RGB')
    img_tensor = eval_tf(img)

    # Predict
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0).to(device))
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()

    real_prob = probs[0].item() * 100
    fake_prob = probs[1].item() * 100
    label     = CLASS_NAMES[pred]
    conf      = max(real_prob, fake_prob)

    # Output
    print()
    print("=" * 40)
    print(f"  Prediction : {label}")
    print(f"  Confidence : {conf:.1f}%")
    print(f"  Real prob  : {real_prob:.1f}%")
    print(f"  Fake prob  : {fake_prob:.1f}%")
    print("=" * 40)

    # Grad-CAM
    if args.gradcam:
        cam         = get_gradcam(model, img_tensor, device, pred)
        base        = os.path.splitext(os.path.basename(args.image))[0]
        output_path = f"{base}_gradcam.png"
        save_gradcam(img, cam, output_path, label)

if __name__ == '__main__':
    main()
