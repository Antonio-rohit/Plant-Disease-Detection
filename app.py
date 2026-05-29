import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from plant_disease_api.main import app  # noqa: E402


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("plant_disease_api.main:app", host="0.0.0.0", port=5000, reload=False)
