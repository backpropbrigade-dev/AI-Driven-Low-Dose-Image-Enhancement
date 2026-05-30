# ClarityDose — AI CT Image Enhancement SaaS

> Low-dose CT reconstruction using Total Variation and Learned Primal-Dual, packaged as a deployable SaaS on GitHub.

[![CI/CD](https://github.com/YOUR_USERNAME/AI-Driven-Low-Dose-Image-Enhancement/actions/workflows/deploy.yml/badge.svg)](https://github.com/YOUR_USERNAME/AI-Driven-Low-Dose-Image-Enhancement/actions)
[![Docker](https://ghcr.io/YOUR_USERNAME/ct-enhancement-api)](https://github.com/YOUR_USERNAME/AI-Driven-Low-Dose-Image-Enhancement/pkgs/container/ct-enhancement-api)

**Live Demo → `https://YOUR_USERNAME.github.io/AI-Driven-Low-Dose-Image-Enhancement`**

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  GitHub Pages (frontend/index.html)             │
│  • Landing page + live demo UI                  │
│  • Drag-and-drop CT image upload                │
│  • Method selector: TV | LPD | Both             │
│  • Side-by-side before/after + metrics          │
└───────────────┬─────────────────────────────────┘
                │ POST /enhance?method=lpd
┌───────────────▼─────────────────────────────────┐
│  FastAPI Backend (backend/main.py)              │
│  • /health   → status check                    │
│  • /enhance  → TV or LPD reconstruction        │
│  Containerised → ghcr.io image                 │
└─────────────────────────────────────────────────┘
```

---

## Quick Start

### Option A — Frontend Only (GitHub Pages, zero cost)

The frontend works in **demo mode** without any backend: it simulates enhancement directly in the browser using Canvas filters. Perfect for showcasing.

1. Fork this repo
2. Go to **Settings → Pages → Source → GitHub Actions**
3. Push to `main` — the GitHub Action deploys `frontend/index.html` automatically
4. Visit `https://YOUR_USERNAME.github.io/AI-Driven-Low-Dose-Image-Enhancement`

### Option B — Full Stack (Frontend + API)

#### Local development

```bash
git clone https://github.com/YOUR_USERNAME/AI-Driven-Low-Dose-Image-Enhancement
cd AI-Driven-Low-Dose-Image-Enhancement

# Start API
docker compose up --build

# Visit http://localhost:8000/docs for the Swagger UI
# Visit frontend/index.html in a browser
```

#### Deploy API to Render (free tier)

1. Create a new **Web Service** on [render.com](https://render.com)
2. Connect your GitHub repo
3. Set **Root Directory** → `backend`
4. Set **Dockerfile** → `Dockerfile`
5. Click **Create Service**
6. Copy the Render URL (e.g. `https://ct-api.onrender.com`)
7. Update the `API_BASE` constant in `frontend/index.html`:
   ```js
   const API_BASE = 'https://ct-api.onrender.com';
   ```
8. Push → GitHub Actions redeploys the frontend automatically

#### Deploy API to Railway

```bash
npm i -g @railway/cli
railway login
railway init
railway up --service backend
```

#### Pull the Docker image

```bash
docker pull ghcr.io/YOUR_USERNAME/ct-enhancement-api:latest
docker run -p 8000:8000 ghcr.io/YOUR_USERNAME/ct-enhancement-api:latest
```

---

## API Reference

### `GET /health`
```json
{ "status": "ok", "version": "1.0.0" }
```

### `POST /enhance`

| Field | Type | Values |
|-------|------|--------|
| `file` | multipart file | PNG, JPEG, TIFF, BMP — max 20 MB |
| `method` | query param | `tv` · `lpd` · `both` |

**Response**
```json
{
  "original": "<base64 PNG>",
  "tv":       "<base64 PNG>",
  "tv_metrics":  { "psnr": 32.4, "ssim": 0.82 },
  "lpd":      "<base64 PNG>",
  "lpd_metrics": { "psnr": 36.1, "ssim": 0.88 }
}
```

---

## GitHub Actions CI/CD

Every push to `main`:

1. **test-backend** — installs deps, smoke-tests imports
2. **build-docker** — builds and pushes to `ghcr.io` (GitHub Container Registry, free)
3. **deploy-frontend** — deploys `frontend/index.html` to GitHub Pages

No secrets needed — uses `GITHUB_TOKEN` (auto-provided).

---

## Results

| Method              | PSNR    | SSIM  |
|---------------------|---------|-------|
| Filtered Back Proj. | ~28 dB  | ~0.72 |
| TV Regularization   | ~32 dB  | ~0.82 |
| Learned Primal-Dual | **~36 dB** | **~0.88** |

Dataset: [LoDoPaB-CT](https://zenodo.org/records/3384092)

---

## Team

**Team Leader:** Anem GnanaGanesh  
**Member:** Annam Yogitha  
**Member:** Chinthamani Manoj Ram Sai

---

## License

MIT
