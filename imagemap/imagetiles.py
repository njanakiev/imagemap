import os
import io
import math
import shutil
import sqlite3
import logging
import itertools
import mercantile
import numpy as np
from PIL import Image
from PIL.Image import DecompressionBombError

EPSILON = 1e-14

logger = logging.getLogger(__name__)



def xy_tile(x, y, zoom):
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


def create_grid_tiles(
    df, 
    folderpath, 
    min_zoom=1,
    max_zoom=10,
    steps=1, 
    tile_size=256,
    remove_folder=False,
    non_geographic=False,
    verbose=False
):
    if os.path.exists(folderpath) and remove_folder:
        shutil.rmtree(folderpath)

    df_tmp = df.copy()
    child_tile_size = tile_size // (2**steps)
    calc_tiles = xy_tile if non_geographic else mercantile.tile

    for zoom in range(min_zoom, max_zoom + 1):
        df_tmp['tile'] = df_tmp.apply(
            lambda row: calc_tiles(row['x'], row['y'], zoom + steps), axis=1)
        df_tiles = df_tmp.drop_duplicates(subset=['tile']).copy()
        
        if verbose:
            print(f"Zoom: {zoom: 3d}, num images: {len(df_tiles): 5d}")
        
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
                    img = img.resize((child_tile_size, child_tile_size))
                    x = row['tile'].x - min_x
                    y = row['tile'].y - min_y
                    img_parent.paste(img, 
                        (x * child_tile_size, y * child_tile_size))
                except DecompressionBombError:
                    print("DecompressionBombError", row['filepath'])
            
            dst_folderpath = os.path.join(
                folderpath, str(parent_tile.z), str(parent_tile.x))
            os.makedirs(dst_folderpath, exist_ok=True)
            dst_filepath = os.path.join(
                dst_folderpath, f"{parent_tile.y}.png")
            
            img_parent.save(dst_filepath)


def create_grid_mbtiles(
    df,
    filepath,
    min_zoom=1,
    max_zoom=10,
    steps=1, 
    tile_size=256,
    image_format='png',
    non_geographic=False,
    verbose=False,
    name='Generated MBTiles',
    description=''
):
    with sqlite3.connect(filepath) as connection:
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
        # TODO: bounds
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
        
        df_tmp = df.copy()
        child_tile_size = tile_size // (2**steps)
        calc_tiles = xy_tile if non_geographic else mercantile.tile
        
        for zoom in range(min_zoom, max_zoom + 1):
            df_tmp['tile'] = df_tmp.apply(
                lambda row: calc_tiles(row['x'], row['y'], zoom + steps), axis=1)
            df_tiles = df_tmp.drop_duplicates(subset=['tile']).copy()

            if verbose:
                print(f"Zoom: {zoom: 3d}, num images: {len(df_tiles): 5d}")
        
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
                        img = img.resize((child_tile_size, child_tile_size))
                        x = row['tile'].x - min_x
                        y = row['tile'].y - min_y
                        img_parent.paste(img, 
                            (x * child_tile_size, y * child_tile_size))
                    except DecompressionBombError:
                        print("DecompressionBombError", row['filepath'])

                if image_format == 'png':
                    image_bytes_format = 'PNG'
                elif image_format == 'jpg':
                    image_bytes_format = 'JPEG'
                else:
                    raise TypeError("Wrong image format: %s", image_format)

                stream = io.BytesIO()
                img_parent.save(stream, format=image_bytes_format)
                image_bytes = stream.getvalue()

                x = parent_tile.x
                y = (2**zoom - 1) - parent_tile.y

                cursor.execute("""
                    INSERT INTO tiles (zoom_level, tile_column, tile_row, tile_data) 
                    VALUES (?, ?, ?, ?);""", 
                    (zoom, x, y, sqlite3.Binary(image_bytes)))
        
        # Optimize database
        #cursor.execute("""ANALYZE;""")
        #cursor.isolation_level = None
        #cursor.execute("""VACUUM;""")
        #cursor.isolation_level = ''
