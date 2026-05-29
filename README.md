# Plant Disease Detection API

A portfolio-ready deep learning project for plant leaf disease classification. The project includes a FastAPI backend, responsive upload UI, TensorFlow/Keras inference, Grad-CAM explainability, Prometheus metrics, tests, Docker support, and CI/CD.

> This project is intended for education, prototyping, and portfolio demonstration. It should not be used for real agricultural decisions without field validation and expert review.

## Highlights

- FastAPI production-style backend
- TensorFlow/Keras image classifier
- MobileNetV2-compatible inference preprocessing
- Secure image upload validation
- Prediction confidence and top-k classes
- Grad-CAM explainability
- Prometheus monitoring endpoint
- Dockerized runtime
- GitHub Actions CI workflow
- Training and evaluation scripts
- MIT licensed

## GitHub-Ready Project Structure

```text
.
|-- app.py
|-- Dockerfile
|-- LICENSE
|-- CONTRIBUTING.md
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|-- .env.example
|-- .dockerignore
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|-- Model/
|   |-- class_indices.json
|   `-- plant_disease_model.h5
|-- docs/
|   `-- MODEL_CARD.md
|-- scripts/
|   |-- train.py
|   `-- evaluate.py
|-- src/
|   `-- plant_disease_api/
|       |-- __init__.py
|       |-- config.py
|       |-- labels.py
|       |-- logging_config.py
|       |-- main.py
|       |-- ml.py
|       |-- rate_limit.py
|       |-- schemas.py
|       `-- security.py
|-- templates/
|   `-- index.html
|-- static/
|   `-- .gitkeep
|-- tests/
|   |-- conftest.py
|   |-- test_api.py
|   |-- test_labels.py
|   `-- test_security.py
`-- notebook/
    `-- Plant_Disease_Detection.ipynb
```

The local `Dataset/`, runtime uploads, generated reports, training artifacts, caches, and duplicate notebook model files are intentionally excluded from Git.

## Quick Start

### 1. Clone and Create Environment

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

### 2. Start the Application

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
uvicorn plant_disease_api.main:app --host 0.0.0.0 --port 5000
```

macOS/Linux:

```bash
export PYTHONPATH=src
uvicorn plant_disease_api.main:app --host 0.0.0.0 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

API docs are available in development mode:

```text
http://127.0.0.1:5000/docs
```

## API Usage

Health check:

```bash
curl http://127.0.0.1:5000/health/live
curl http://127.0.0.1:5000/health/ready
```

Prediction:

```bash
curl -X POST -F "file=@leaf.jpg" -F "explain=true" http://127.0.0.1:5000/api/v1/predict
```

Classes:

```bash
curl http://127.0.0.1:5000/api/v1/classes
```

Metrics:

```bash
curl http://127.0.0.1:5000/metrics
```

## Dataset Setup

The training pipeline expects this layout:

```text
Dataset/
`-- New Plant Diseases Dataset(Augmented)/
    `-- New Plant Diseases Dataset(Augmented)/
        |-- train/
        `-- valid/
```

The dataset is intentionally not committed because it is large. Download it from Kaggle and place it under `Dataset/` using the layout above.

## Train the Model

```bash
python scripts/train.py ^
  --data-dir "Dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)" ^
  --output-dir artifacts ^
  --backbone efficientnetv2b0 ^
  --epochs 30
```

macOS/Linux:

```bash
python scripts/train.py \
  --data-dir "Dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)" \
  --output-dir artifacts \
  --backbone efficientnetv2b0 \
  --epochs 30
```

Supported backbones:

- `efficientnetv2b0`
- `mobilenetv2`
- `resnet50`

The script saves the model, training history, TensorBoard logs, and `class_indices.json` in the output folder.

## Evaluate the Model

```bash
python scripts/evaluate.py ^
  --model Model/plant_disease_model.h5 ^
  --data "Dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid" ^
  --preprocessor mobilenetv2
```

Metrics are saved to `reports/evaluation.json` and include accuracy, precision, recall, F1 score, ROC-AUC when computable, and confusion matrix.

## Run Tests

```bash
pytest
```

Syntax check:

```bash
python -m compileall app.py src scripts tests
```

## Docker

Build:

```bash
docker build -t plant-disease-detection .
```

Run:

```bash
docker run --rm -p 5000:5000 plant-disease-detection
```

## Configuration

Copy `.env.example` to `.env` for local experimentation. Do not commit `.env`.

Key variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PLANT_MODEL_PATH` | `Model/plant_disease_model.h5` | Model artifact path |
| `PLANT_CLASS_INDEX_PATH` | `Model/class_indices.json` | Class label mapping |
| `UPLOAD_DIR` | `static/uploads` | Runtime upload folder |
| `MAX_UPLOAD_MB` | `8` | Upload size limit |
| `TOP_K_PREDICTIONS` | `3` | Number of prediction alternatives |
| `RATE_LIMIT_PER_MINUTE` | `30` | Basic per-client abuse protection |
| `ENABLE_GRADCAM` | `true` | Grad-CAM generation toggle |
| `ENABLE_DOCS` | `true` | FastAPI docs toggle |

## Security Notes

Implemented safeguards:

- Extension allowlist
- Image content verification
- Upload size limit
- Randomized safe filenames
- Security response headers
- Rate limiting
- Non-root Docker runtime
- Ignored runtime uploads and local secrets

For a public deployment, add API authentication, gateway-level rate limiting, CORS origin restrictions, upload retention policies, and dependency vulnerability scanning.

## Deployment Recommendations

Recommended beginner-friendly options:

- Render web service with Docker
- Railway Docker deployment
- Google Cloud Run
- AWS App Runner
- Azure Container Apps

Production-grade path:

1. Store model artifacts in object storage or a model registry.
2. Build and scan the Docker image in CI.
3. Deploy behind HTTPS with an API gateway.
4. Export Prometheus metrics to a monitoring stack.
5. Add application logs, prediction logs, model version tracking, and drift monitoring.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
