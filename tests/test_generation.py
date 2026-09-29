import modal

from src.generation import build_image, build_test_image


def test_build_image_returns_modal_image():
    image = build_image()

    assert isinstance(image, modal.Image)


def test_build_test_image_returns_modal_image():
    image = build_test_image()

    assert isinstance(image, modal.Image)
