import os
import math
import shutil
import itertools
import mercantile
import numpy as np
from PIL import Image
from PIL.Image import DecompressionBombError

EPSILON = 1e-14


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


def create_grid_tiles(df, 
                      folderpath, 
                      min_zoom=1, max_zoom=10, steps=1, 
                      tile_size=256,
                      remove_folder=False,
                      non_geographic=False,
                      verbose=False):

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
                    img_parent.paste(img, (x * child_tile_size, y * child_tile_size))
                except DecompressionBombError:
                    print("DecompressionBombError", row['filepath'])
            
            dst_folderpath = os.path.join(folderpath, str(parent_tile.z), str(parent_tile.x))
            os.makedirs(dst_folderpath, exist_ok=True)
            dst_filepath = os.path.join(dst_folderpath, f"{parent_tile.y}.png")
            img_parent.save(dst_filepath)
