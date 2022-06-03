import os
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

from . import utils


def resize_images(src_folder, dst_folder, size, overwrite=False):
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)

    for filename in os.listdir(src_folder):
        src_filepath = os.path.join(src_folder, filename)
        dst_filepath = os.path.join(dst_folder, filename)
        print(src_filepath)

        if not os.path.exists(dst_filepath) or overwrite:
            image = Image.open(src_filepath)
            try:
                exif_bytes = image.info["exif"]
                ImageOps.fit(image, size, Image.ANTIALIAS).save(
                    dst_filepath, exif=exif_bytes)
            except Exception as e:
                print(e)
                break


def images_gps_metadata(folderpath):
    items = []
    for filename in os.listdir(folderpath):
        if not filename.lower().endswith('jpg'):
            continue

        try:
            filepath = os.path.join(folderpath, filename)
            item = utils.get_exif_metadata(filepath)
            item['filepath'] = filepath
            items.append(item)
        except UnidentifiedImageError as e:
            print("UnidentifiedImageError", filepath)

    df = pd.DataFrame(items)
    df[['lat', 'lon']] = df['GPSInfo'].apply(
        lambda d: pd.Series(utils.get_lat_lon(d), index=['lat', 'lon']))

    return df


if __name__ == '__main__':
    src_folder = "/mnt/volume/personal-data/Camera/"
    dst_folder = "/mnt/volume/personal-data/Camera_resized"
    size = (300, 300)

    #resize_images(src_folder, dst_folder, size)

    gdf_images = images_gps_metadata(dst_folder)
    gdf_images[['DateTime', 'filepath', 'geom']].to_file(
        'photos.geojson', driver='GeoJSON')
