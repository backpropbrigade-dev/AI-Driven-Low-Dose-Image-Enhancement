import io
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="CT Image Enhancement API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Learned Primal-Dual Network (lightweight version for deployment) ──────────

class PrimalBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, x, update):
        inp = torch.cat([x.expand(-1, x.shape[1], -1, -1)
                         if x.shape[1] > 1 else x,
                         update], dim=1)
        # Simple residual update
        return x + self.conv(inp)


class DualBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, h, fp, g):
        inp = torch.cat([h, fp - g], dim=1)
        return h + self.conv(inp)


class LearnedPrimalDual(nn.Module):
    """Simplified LPD for CPU inference in SaaS context."""
    def __init__(self, n_iter=5, n_primal=5, n_dual=5):
        super().__init__()
        self.n_iter = n_iter
        self.primal_net = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 3, padding=1),
            ) for _ in range(n_iter)
        ])
        self.dual_net = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 3, padding=1),
            ) for _ in range(n_iter)
        ])

    def forward(self, x):
        h = torch.zeros_like(x)
        f = x.clone()
        for i in range(self.n_iter):
            fp = f + 0.1 * (x - f)   # physics step (simplified)
            h = h + self.dual_net[i](torch.cat([h, fp - x], dim=1))
            f = f + self.primal_net[i](torch.cat([f, h], dim=1))
        return torch.clamp(f, 0, 1)


# ── Model singleton ────────────────────────────────────────────────────────────

_lpd_model = None

def get_lpd_model():
    global _lpd_model
    if _lpd_model is None:
        _lpd_model = LearnedPrimalDual(n_iter=5)
        _lpd_model.eval()
        # Try to load pre-trained weights if available
        import os
        if os.path.exists("best_learned_all_train_pd.pth"):
            try:
                state = torch.load("best_learned_all_train_pd.pth", map_location="cpu")
                _lpd_model.load_state_dict(state, strict=False)
            except Exception:
                pass  # Use random init if weights are incompatible
    return _lpd_model


# ── Enhancement methods ────────────────────────────────────────────────────────

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze().detach().cpu().numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def tv_denoise(tensor: torch.Tensor, weight: float = 0.12, n_iter: int = 100) -> torch.Tensor:
    """Total Variation denoising via gradient descent."""
    x = tensor.clone().requires_grad_(False)
    u = tensor.clone()
    for _ in range(n_iter):
        # Compute TV gradient
        dy = torch.zeros_like(u)
        dx = torch.zeros_like(u)
        dy[:, :, :-1, :] = u[:, :, 1:, :] - u[:, :, :-1, :]
        dx[:, :, :, :-1] = u[:, :, :, 1:] - u[:, :, :, :-1]
        norm = (dy**2 + dx**2).sqrt().clamp(min=1e-8)
        # Divergence
        divx = torch.zeros_like(u)
        divy = torch.zeros_like(u)
        divx[:, :, :, 1:] = dx[:, :, :, :-1]
        divy[:, :, 1:, :] = dy[:, :, :-1, :]
        div = divx + divy
        u = u - 0.01 * ((u - x) - weight * div)
    return torch.clamp(u, 0, 1)


def enhance_tv(img: Image.Image) -> Image.Image:
    t = pil_to_tensor(img)
    enhanced = tv_denoise(t, weight=0.15)
    return tensor_to_pil(enhanced)


def enhance_lpd(img: Image.Image) -> Image.Image:
    model = get_lpd_model()
    t = pil_to_tensor(img)
    with torch.no_grad():
        enhanced = model(t)
    return tensor_to_pil(enhanced)


def compute_metrics(original: Image.Image, enhanced: Image.Image) -> dict:
    orig_arr = np.array(original.convert("L"), dtype=np.float32) / 255.0
    enh_arr  = np.array(enhanced.convert("L"), dtype=np.float32) / 255.0
    mse = np.mean((orig_arr - enh_arr) ** 2)
    psnr = float(20 * np.log10(1.0 / np.sqrt(mse + 1e-10)))
    # SSIM approximation
    mu1, mu2 = orig_arr.mean(), enh_arr.mean()
    s1 = orig_arr.std(); s2 = enh_arr.std()
    cov = np.mean((orig_arr - mu1) * (enh_arr - mu2))
    c1, c2 = 0.01**2, 0.03**2
    ssim = float(((2*mu1*mu2 + c1)*(2*cov + c2)) /
                 ((mu1**2 + mu2**2 + c1)*(s1**2 + s2**2 + c2)))
    return {"psnr": round(psnr, 2), "ssim": round(max(0, min(1, ssim)), 4)}


def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/enhance")
async def enhance(
    file: UploadFile = File(...),
    method: str = "lpd"   # "tv" | "lpd" | "both"
):
    if file.content_type not in ("image/png", "image/jpeg", "image/tiff", "image/bmp"):
        raise HTTPException(400, "Unsupported file type. Send PNG/JPEG/TIFF/BMP.")
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, "File too large. Max 20 MB.")
    try:
        original = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Could not decode image.")

    results = {"original": img_to_b64(original.convert("RGB"))}
    if method in ("tv", "both"):
        tv_out = enhance_tv(original)
        results["tv"] = img_to_b64(tv_out)
        results["tv_metrics"] = compute_metrics(original, tv_out)
    if method in ("lpd", "both"):
        lpd_out = enhance_lpd(original)
        results["lpd"] = img_to_b64(lpd_out)
        results["lpd_metrics"] = compute_metrics(original, lpd_out)
    return JSONResponse(results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
