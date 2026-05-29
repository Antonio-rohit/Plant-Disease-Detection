import re
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError


SAFE_STEM_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def sanitize_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = SAFE_STEM_RE.sub("-", stem).strip(".-")
    return stem[:80] or "leaf"


def extension_for(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


async def save_validated_upload(
    upload: UploadFile,
    upload_dir: Path,
    allowed_extensions: frozenset[str],
    max_upload_mb: int,
) -> Path:
    if not upload.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No image file was uploaded.")

    extension = extension_for(upload.filename)
    if extension not in allowed_extensions:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Unsupported file type. Upload a JPG, JPEG, PNG, or WEBP image.",
        )

    max_bytes = max_upload_mb * 1024 * 1024
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {max_upload_mb} MB.")
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty.")

    filename = f"{sanitize_stem(upload.filename)}_{uuid4().hex}.{extension}"
    destination = upload_dir / filename
    destination.write_bytes(data)

    try:
        with Image.open(destination) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is not a valid image.") from exc

    return destination
