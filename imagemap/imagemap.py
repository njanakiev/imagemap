import logging
import numpy as np
from PIL import Image, ImageOps
from PIL.Image import DecompressionBombError
from . import utils

logger = logging.getLogger(__name__)


def create_image(
    df,
    size,
    extent=None,
    image_size=256,
    gridded=False,
    square_images=False,
    margin=0,
    background_color=(255, 255, 255, 0),
    verbose=False
):
    width, height = size
    
    if extent is None:
        extent = np.concatenate([
            df[['x', 'y']].values.min(axis=0),
            df[['x', 'y']].values.max(axis=0)
        ])

    outer_extent = utils.scale_extent(
        extent, width, height, boundary_type='outer')

    min_x, min_y, max_x, max_y = outer_extent
    n = width  // image_size
    m = height // image_size

    full_image = Image.new("RGBA", (width, height), background_color)

    for idx, row in df.reset_index().iterrows():
        x, y = row['x'], row['y']
        filepath = row['filepath']

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
            img = Image.open(filepath)
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
            logging.warn("DecompressionBombError", filepath)

    return full_image, outer_extent
