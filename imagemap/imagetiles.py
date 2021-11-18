import os
import io
import math
import shutil
import sqlite3
import logging
import itertools
import mercantile
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from PIL.Image import DecompressionBombError

from . import imagemap
from . import utils

EPSILON = 1e-14
FULL_EXTENT = [
    -180.0, -85.0511287798066, 
     180.0,  85.0511287798066
]
FULL_EXTENT_XY = [
    -20037508.342789244, -20037508.342789244, 
     20037508.342789244,  20037508.342789244
]

logger = logging.getLogger(__name__)


def xy_tile(x, y, zoom):
    """Get tile for normalized x, y coordinates (between 0 and 1) for a specific zoom level"""

    Z2 = math.pow(2, zoom)
    y = 1 - y  # invert
    
    if x <= 0:
        xtile = 0
    elif x >= 1:
        xtile = int(Z2 - 1)
    else:
        xtile = int(math.floor((x + EPSILON) * Z2))

    if y <= 0:
        ytile = 0
    elif y >= 1:
        ytile = int(Z2 - 1)
    else:
        ytile = int(math.floor((y + EPSILON) * Z2))

    return mercantile.Tile(xtile, ytile, zoom)


def generate_gridded_tiles(
    df,
    min_zoom=0,
    max_zoom=6,
    steps=1,
    tile_size=256,
    non_geographic=False,
    verbose=True
):
    df_tmp = df[['x', 'y', 'filepath']].copy()
    
    child_tile_size = tile_size // (2**steps)
    calc_tiles = xy_tile if non_geographic else mercantile.tile
    
    for zoom in range(min_zoom, max_zoom + 1):
        df_tmp['tile'] = df_tmp.apply(
            lambda row: calc_tiles(row['x'], row['y'], zoom + steps), axis=1)
        
        # Take the first unique tile
        df_tiles = df_tmp.drop_duplicates(subset=['tile']).copy()

        if verbose:
            logger.info(f"Zoom: {zoom: 2d}, num images: {len(df_tiles): 5d}")
    
        df_tiles['parent_tile'] = df_tiles.apply(
            lambda row: mercantile.parent(row['tile'], zoom=row['tile'].z - steps), 
            axis=1)
        
        for parent_tile, group in df_tiles.groupby('parent_tile'):
            children = [parent_tile]
            for _ in range(steps):
                children = [mercantile.children(tile) for tile in children]
                children = itertools.chain.from_iterable(children)
            
            X = np.array([(child.x, child.y) for child in children])
            min_x, min_y = X.min(axis=0)

            img_parent = Image.new("RGBA", (tile_size, tile_size))
            for _, row in group.iterrows():
                try:
                    img = Image.open(row['filepath'])
                    img = ImageOps.fit(img, (child_tile_size, child_tile_size))
                    x = row['tile'].x - min_x
                    y = row['tile'].y - min_y
                    img_parent.paste(img, 
                        (x * child_tile_size, y * child_tile_size))
                except DecompressionBombError:
                    logger.warn(f"DecompressionBombError {row['filepath']}")
        
            yield img_parent, parent_tile.x, parent_tile.y, zoom


def generate_tiles(
    df,
    image_size,
    min_zoom=0,
    max_zoom=6,
    non_geographic=False,
    verbose=True
):
    df_tmp = df[['x', 'y', 'filepath']].copy().dropna()

    if non_geographic:
        # Transform from (0, 1) to full extent
        df_tmp['x'] = FULL_EXTENT_XY[0] + \
            (df_tmp['x'] * (FULL_EXTENT_XY[2] - FULL_EXTENT_XY[0]))
        df_tmp['y'] = FULL_EXTENT_XY[1] + \
            (df_tmp['y'] * (FULL_EXTENT_XY[3] - FULL_EXTENT_XY[1]))
    else: 
        df_tmp[['x', 'y']] = df_tmp.apply(
            lambda row: pd.Series(
                mercantile.xy(row['x'], row['y'], truncate=True),
                index=['x', 'y']),
            axis=1)

    extent_xy = np.concatenate([
        df_tmp[['x', 'y']].values.min(axis=0),
        df_tmp[['x', 'y']].values.max(axis=0)
    ])

    extent = np.concatenate([
        mercantile.lnglat(extent_xy[0], extent_xy[1]),
        mercantile.lnglat(extent_xy[2], extent_xy[3])
    ])

    for zoom in range(min_zoom, max_zoom + 1):
        tile_size = 256 * (2**zoom)

        if verbose:
            logger.info(f"Zoom: {zoom: 2d}")

        _, size = utils.relative_extent(
            FULL_EXTENT_XY, extent_xy, (tile_size, tile_size))

        full_image, image_extent = imagemap.create_image(
            df, 
            size,
            extent=extent_xy,
            image_size=image_size,
            verbose=verbose
        )

        for tile in mercantile.tiles(*extent, zoom):
            tile_extent = list(mercantile.xy_bounds(tile))

            crop, _ = utils.relative_extent(
                extent_xy, 
                tile_extent, 
                full_image.size, 
                invert_y=True
            )
            img = full_image.crop(crop)

            yield img, tile.x, tile.y, zoom


def create_grid_tiles(
    df,
    dst_filepath,
    output_type='folder',
    min_zoom=1,
    max_zoom=10,
    steps=1, 
    tile_size=256,
    image_size=20,
    image_format='png',
    gridded_tiles=True,
    non_geographic=False,
    overwrite=False,
    verbose=False,
    name='Generated Tiles',
    description=''
):
    if len(df) == 0:
        raise ValueError("Dataframe df has no columns!")
    
    if gridded_tiles:
        tiles_generator = generate_gridded_tiles(
            df,
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            steps=steps,
            tile_size=tile_size,
            non_geographic=non_geographic,
            verbose=verbose
        )
    else:
        tiles_generator = generate_tiles(
            df,
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            image_size=image_size,
            non_geographic=non_geographic,
            verbose=verbose
        )

    if image_format == 'png':
        image_bytes_format = 'PNG'
    elif image_format == 'jpg':
        image_bytes_format = 'JPEG'
    else:
        raise TypeError(f"Wrong image format: {image_format}")
    
    if output_type.lower() == 'folder':
        if os.path.exists(dst_filepath) and overwrite:
            shutil.rmtree(dst_filepath)

        for img, x, y, zoom in tiles_generator:
            dst_folderpath = os.path.join(dst_filepath, str(zoom), str(x))
            os.makedirs(dst_folderpath, exist_ok=True)
            image_filepath = os.path.join(dst_folderpath, f"{y}.png")  
            img.save(image_filepath, format=image_bytes_format)

    elif output_type.lower() == 'mbtiles':
        if os.path.exists(dst_filepath) and overwrite:
            os.remove(dst_filepath)
        
        with sqlite3.connect(dst_filepath) as connection:
            cursor = connection.cursor()
            
            # Initialize database
            #cursor.execute("""PRAGMA synchronous=0""")
            #cursor.execute("""PRAGMA locking_mode=EXCLUSIVE""")
            #cursor.execute("""PRAGMA journal_mode=DELETE""")
            cursor.execute("""
                CREATE TABLE tiles (
                    zoom_level INTEGER,
                    tile_column INTEGER,
                    tile_row INTEGER,
                    tile_data BLOB);""")
            cursor.execute("""
                CREATE TABLE metadata(
                    name TEXT, 
                    value TEXT);""")
            cursor.execute("""
                CREATE UNIQUE INDEX name ON metadata (name);""")
            cursor.execute("""
                CREATE UNIQUE INDEX tile_index ON tiles
                    (zoom_level, tile_column, tile_row);""")
            
            # Fill metadata
            # TODO: bounds metadata
            metadata = [
                ('name', name),
                ('descrption', description),
                ('version', '1.1'),
                ('format', image_format),
                ('type', 'overlay'),
                ('minzoom', min_zoom),
                ('maxzoom', max_zoom)
            ]
            for (name, value) in metadata:
                cursor.execute("""
                    INSERT INTO metadata (name, value) VALUES (?, ?)""",
                    (name, value))
            
            for img, x, y, zoom in tiles_generator:
                y = (2**zoom - 1) - y
                stream = io.BytesIO()
                img.save(stream, format=image_bytes_format)
                image_bytes = stream.getvalue()
                cursor.execute("""
                    INSERT INTO tiles (zoom_level, tile_column, tile_row, tile_data) 
                    VALUES (?, ?, ?, ?);""", 
                    (zoom, x, y, sqlite3.Binary(image_bytes)))
            
            # Optimize database
            #cursor.execute("""ANALYZE;""")
            #cursor.isolation_level = None
            #cursor.execute("""VACUUM;""")
            #cursor.isolation_level = ''
    else:
        raise TypeError(f"Wrong output type: {output_type}")
