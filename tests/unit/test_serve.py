"""
Unit tests for src/serve.py validate_image — L11 contracts.

Tests (4):

  U33: Valid PNG passes validate_image → RGB PIL Image

  U34: < 32px raises HTTPException 422

  U35: Blank image raises HTTPException 422

  U36: Non-image bytes raise HTTPException 422 (magic byte check)

"""

import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.unit


def test_u33_validate_image_valid(synthetic_png_bytes):
    """U33: Valid PNG passes validate_image and returns RGB PIL Image."""
    from PIL import Image

    from src.serve import validate_image

    result = validate_image(synthetic_png_bytes)

    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"


def test_u34_validate_image_too_small(tiny_png_bytes):
    """U34: Image < 32px raises HTTPException 422."""
    from src.serve import validate_image

    with pytest.raises(HTTPException) as exc:
        validate_image(tiny_png_bytes)

    assert exc.value.status_code == 422


def test_u35_validate_image_blank(blank_png_bytes):
    """U35: Blank/uniform image raises HTTPException 422."""
    from src.serve import validate_image

    with pytest.raises(HTTPException) as exc:
        validate_image(blank_png_bytes)

    assert exc.value.status_code == 422


def test_u36_validate_image_non_image():
    """U36: Non-image bytes raise HTTPException 422 (magic byte validation)."""
    from src.serve import validate_image

    with pytest.raises(HTTPException) as exc:
        validate_image(b"This is plain text, not an image.")

    assert exc.value.status_code == 422
