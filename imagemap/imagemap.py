import logging
import requests
import numpy as np
from PIL import Image, ImageOps
from PIL.Image import DecompressionBombError, UnidentifiedImageError
from . import utils

logger = logging.getLogger(__name__)


def _download_image(url):
    r = requests.get(url, stream=True)
    r.raise_for_status()

    return Image.open(r.raw)


def _get_loader(image_type):
    if image_type == 'url':
        loader = _download_image
    elif image_type == 'filepath':
        loader = Image.open
    elif image_type == 'pil':
        loader = lambda im: im
    else:
        raise ValueError(
            f"Image type not available: {image_type}")

    return loader


def image_grid(
    images,
    nrows, ncols,
    tile_size=128,
    padding=0,
    image_type='filepath'
):
    loader = _get_loader(image_type)
    w = ncols * (tile_size + padding) + padding
    h = nrows * (tile_size + padding) + padding
    grid_image = Image.new(
        'RGB', (w, h), (255, 255, 255))

    for idx in range(nrows*ncols):
        if idx < len(images):
            #img = Image.open(images[idx]) if image_paths else images[idx]
            img = loader(images[idx])
            img = img.convert('RGB')
            img_square = ImageOps.fit(img, (tile_size, tile_size))
            img_square = img_square.resize(
                (tile_size, tile_size))

            i = idx % ncols
            j = (idx - i) // ncols
            offset = (
                (tile_size + padding) * i + padding,
                (tile_size + padding) * j + padding
            )
            grid_image.paste(img_square, offset)

    return grid_image



def image_map(
    images,
    X,
    size,
    extent=None,
    image_size=256,
    gridded=False,
    square_images=False,
    margin=0,
    background_color=(255, 255, 255, 0),
    verbose=False,
    image_type='filepath'
):
    loader = _get_loader(image_type)
    width, height = size

    if extent is None:
        extent = np.concatenate([X.min(axis=0), X.max(axis=0)])

    outer_extent = utils.scale_extent(
        extent, width, height, boundary_type='outer')

    min_x, min_y, max_x, max_y = outer_extent
    n = width  // image_size
    m = height // image_size

    full_image = Image.new("RGBA", (width, height), background_color)

    #for idx, row in df.reset_index().iterrows():
    for idx, ((x, y), image) in enumerate(zip(X, images)):
        if verbose and idx % 500 == 0:
            logger.info(f"Num images: {idx}")
        
        # Normalize between 0 and 1
        x = (x - min_x) / (max_x - min_x)
        y = (y - min_y) / (max_y - min_y)

        if gridded:
            x = image_size * int(x * n + 0.5)
            y = height - image_size * int(y * m + 0.5)
        else:
            #x = int(x * width - (image_size // 2))
            #y = height - int(y * height + (image_size // 2))
            x = margin + int(x * (width - image_size - 2 * margin))
            y = height - image_size - margin \
                       - int(y * (height - image_size - 2 * margin))

        try:
            img = loader(image)
            if square_images or gridded:
                img = ImageOps.fit(
                    img, (image_size, image_size), Image.ANTIALIAS)
            else:
                img.thumbnail(
                    (image_size, image_size), Image.ANTIALIAS)

            if len(img.mode) == 4:
                full_image.paste(img, (x, y), img)
            else:
                full_image.paste(img, (x, y))
        except DecompressionBombError:
            path = image if image_type != "pil" else "IMAGE"
            logging.warn(f"[DecompressionBombError] for {path}")
        except UnidentifiedImageError:
            path = image if image_type != "pil" else "IMAGE"
            logging.warn(f"[UnidentifiedImageError] for {path}")
        except requests.exceptions.HTTPError as e:
            logging.warn(f"[HTTPError] {e} for {image}")

    return full_image, outer_extent
