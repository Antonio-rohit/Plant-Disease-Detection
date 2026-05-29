import json
from pathlib import Path


def load_class_names(class_index_path: Path, class_dir: Path) -> list[str]:
    if class_index_path.exists():
        class_indices = json.loads(class_index_path.read_text(encoding="utf-8"))
        return [name for name, _ in sorted(class_indices.items(), key=lambda item: item[1])]

    if not class_dir.exists():
        raise FileNotFoundError(
            "Class labels not found. Provide Model/class_indices.json or set PLANT_CLASS_DIR."
        )

    class_names = sorted(
        item.name for item in class_dir.iterdir() if item.is_dir() and not item.name.startswith(".")
    )
    if not class_names:
        raise RuntimeError(f"No class folders found in {class_dir}")
    return class_names


def format_class_name(class_name: str) -> str:
    return class_name.replace("___", " : ").replace("_", " ")
