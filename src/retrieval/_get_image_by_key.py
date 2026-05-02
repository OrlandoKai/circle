import io
import os
import tarfile
from pathlib import Path

from PIL import Image


def get_image_by_key(key: str, root: str | Path, from_tar: bool = True) -> Image.Image | None:
    """Get an image by its key.

    Args:
    ----
        key (str): Image key.
        root (str): Root directory where the tar files are stored.
        from_tar (bool): Whether to read from tar files. Default is True.

    """
    assert key is not None and len(key) > 5, "Key must be at least 5 characters long"

    tar_key = key[:5]
    corresponding_folder = os.path.join(root, f"{tar_key}")

    image_original = None

    if from_tar:
        with tarfile.open(corresponding_folder + ".tar") as tar:
            if image_data := tar.extractfile(f"{key}.jpg"):
                image_data = image_data.read()
                image_original = Image.open(io.BytesIO(image_data)).convert("RGB")

    else:
        image_path = os.path.join(corresponding_folder, f"{key}.jpg")
        if os.path.exists(image_path):
            image_original = Image.open(image_path).convert("RGB")

    return image_original


def get_images_by_key(
    keys: list[str], root: str | Path, from_tar: bool = True
) -> dict[str, Image.Image | None]:
    """Get images by their keys.

    Args:
    ----
        keys (list[str]): List of image keys.
        root (str): Root directory where the tar files are stored.
        from_tar (bool): Whether to read from tar files. Default is True.

    """
    assert keys is not None and all(
        key is not None and len(key) > 5 for key in keys
    ), "All keys must be at least 5 characters long"

    all_images = {}

    # Group images by tar to perform batch read
    if from_tar:
        img2tar = {}

        for x in keys:
            prefix = x[:5]
            if prefix not in img2tar:
                img2tar[prefix] = []

            img2tar[prefix].append(x)

        # Iterate over tar files
        for tar_key, images in img2tar.items():
            corresponding_folder = os.path.join(root, f"{tar_key}")

            with tarfile.open(corresponding_folder + ".tar") as tar:
                for key in images:
                    if image_data := tar.extractfile(f"{key}.jpg"):
                        image_data = image_data.read()
                        image_original = Image.open(io.BytesIO(image_data)).convert("RGB")
                    else:
                        image_original = None
                    all_images[key] = image_original

    else:
        for key in keys:
            image_path = os.path.join(root, key[:5], f"{key}.jpg")
            if os.path.exists(image_path):
                try:
                    image_original = Image.open(image_path).convert("RGB")
                except Exception:
                    image_original = None
            else:
                image_original = None
            all_images[key] = image_original

    return all_images
