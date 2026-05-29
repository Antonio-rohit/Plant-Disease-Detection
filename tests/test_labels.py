import json

from plant_disease_api.labels import format_class_name, load_class_names


def test_load_class_names_prefers_index_file(tmp_path):
    class_index = tmp_path / "class_indices.json"
    class_index.write_text(json.dumps({"b___Disease": 1, "a___Healthy": 0}), encoding="utf-8")

    assert load_class_names(class_index, tmp_path / "missing") == ["a___Healthy", "b___Disease"]


def test_load_class_names_falls_back_to_directory(tmp_path):
    class_dir = tmp_path / "train"
    (class_dir / "Tomato___healthy").mkdir(parents=True)
    (class_dir / "Apple___Apple_scab").mkdir(parents=True)

    assert load_class_names(tmp_path / "missing.json", class_dir) == [
        "Apple___Apple_scab",
        "Tomato___healthy",
    ]


def test_format_class_name():
    assert format_class_name("Tomato___Late_blight") == "Tomato : Late blight"
