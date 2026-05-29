from plant_disease_api.security import sanitize_stem


def test_sanitize_stem_removes_path_and_unsafe_chars():
    assert sanitize_stem("../../bad name<script>.jpg") == "bad-name-script"


def test_sanitize_stem_has_default():
    assert sanitize_stem("###.png") == "leaf"
