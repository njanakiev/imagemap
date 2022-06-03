import os
import shapely.geometry
import shapely.ops

import rasterio as rio
import rasterio.merge
import rasterio.mask
import rasterio.plot
from rasterio.enums import Resampling

from . import utils


def georeference_images(dst_folder, gdf, bounds, depth=4):
    bounds_box = shapely.geometry.box(*bounds)
    mask = gdf.within(bounds_box)
    output_filepaths = []
    
    if mask.sum() == 1 or (mask.sum() > 1 and depth == 0):
        src_filepath = gdf[mask].iloc[0]['filepath']
        filename = os.path.splitext(os.path.basename(src_filepath))[0]
        dst_filepath = os.path.join(dst_folder, filename + '.tiff')
        try:
            with rio.open(src_filepath, 'r') as src_image:
                profile = {
                    'driver': 'Gtiff',
                    'dtype': 'uint8',
                    'width': src_image.width,
                    'height': src_image.height,
                    'count': 3,
                    'crs': rio.crs.CRS.from_epsg(3785),
                    'transform': rio.transform.from_bounds(
                        *bounds, src_image.width, src_image.height),
                    'nodata': 0
                }
                with rio.open(dst_filepath, 'w', **profile) as dst_image:
                    for i in range(1, 4):
                        band = src_image.read(i)
                        band[band == 0] = 1
                        dst_image.write(band, i)

            return [dst_filepath]
        except Exception as e:
            print('EXCEPTION:', e)
            print(src_filepath)
            return []
            
    elif mask.sum() > 1:
        for b in utils.quad_rectangle(bounds):
            output_filepaths.extend(georeference_images(
                dst_folder, gdf[mask], b, depth - 1))
            
    return output_filepaths


def merge_images(filepaths, dst_filepath):
    files = [rio.open(f, 'r') for f in filepaths]
    
    mosaic, out_trans = rasterio.merge.merge(files, nodata=0)
    profile = {
        'driver': 'Gtiff',
        'dtype': 'uint8',
        'width': mosaic.shape[2],
        'height': mosaic.shape[1],
        'count': 3,
        'crs': rio.crs.CRS.from_epsg(3785),
        'transform': out_trans,
        'nodata': 0
    }
    
    with rio.open(dst_filepath, 'w', **profile) as geotiff:
        geotiff.write(mosaic)


def crop(src_filepath, dst_filepath, geom):
    with rasterio.open(src_filepath, mode='r') as src:
        bbox = utils.scale_extent(geom.bounds, src.shape[1], src.shape[0])
        
        out_meta = {
            "driver": "GTiff",
            "height": src.shape[0],
            "width": src.shape[1],
            "transform": rio.transform.from_bounds(*bbox, src.width, src.height),
            "nodata": 0,
            "count": 3,
            "dtype": "uint8",
            "crs": rio.crs.CRS.from_epsg(3857)
        }
        with rio.MemoryFile() as memfile:
            with memfile.open(**out_meta) as dst:  # Open as DatasetWriter
                for i in range(1, 4):
                    band = src.read(i)
                    band[band == 0] = 1
                    dst.write(band, i)

            with memfile.open() as src:  # Reopen as DatasetReader
                out_image, out_transform = rasterio.mask.mask(src, [geom], crop=True)
                
                rasterio.plot.show(out_image)
                
                out_meta = {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": 0,
                    "count": 3,
                    "dtype": "uint8",
                    "crs": rio.crs.CRS.from_epsg(3857)
                }
                with rasterio.open(dst_filepath, "w", **out_meta) as dst:
                    dst.write(out_image)
        
        
def crop_and_resample(src_filepath, dst_filepath, geom, scale_factor=1.0, dst_size=None, epsg=3857):
    """Crop and resample input image to GeoTiff output image"""
    
    with rasterio.open(src_filepath, mode='r+') as src:
        bbox = utils.scale_extent(geom.bounds, src.shape[1], src.shape[0])
        src.transform = rio.transform.from_bounds(*bbox, src.width, src.height)
        
        if dst_size is None:
            dst_size = (
                #int(src.height * scale_factor),
                int(src.width * scale_factor),
                int(src.height * scale_factor)
            )
            
        out_shape = (src.count, dst_size[0], dst_size[1])
        out_image = src.read(
            out_shape=out_shape, 
            resampling=Resampling.bilinear
        )
        
        # Make all 0 values to 1
        out_image[out_image == 0] = 1
        
        # scale image transform
        out_transform = src.transform * src.transform.scale(
            (src.width / out_image.shape[-1]),
            (src.height / out_image.shape[-2])
        )
        
        out_meta = {
            "driver": "GTiff",
            "height": out_shape[1],
            "width": out_shape[2],
            "transform": out_transform,
            "nodata": 0,
            "count": 3,
            "dtype": "uint8",
            "crs": rio.crs.CRS.from_epsg(epsg)
        }
        
        with rio.MemoryFile() as memfile:
            with memfile.open(**out_meta) as dst:  # Open as DatasetWriter
                dst.write(out_image)

            with memfile.open() as src:  # Reopen as DatasetReader
                out_image, out_transform = rasterio.mask.mask(src, [geom], crop=True)
                
                out_meta = {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": 0,
                    "count": 3,
                    "dtype": "uint8",
                    "crs": rio.crs.CRS.from_epsg(epsg)
                }
                with rasterio.open(dst_filepath, "w", **out_meta) as dst:
                    dst.write(out_image)


def crop_and_resample_collection(gdf, dst_folderpath, scale_factor=0.2, epsg=3857):
    if not os.path.exists(dst_folderpath):
        os.mkdir(dst_folderpath)
    
    for idx, row in gdf.to_crs(epsg=epsg).iterrows():
        src_filepath, geom = row['filepath'], row['geom']
        
        filename = os.path.basename(src_filepath).replace('.jpg', '.tif')
        dst_filepath = os.path.join(dst_folderpath, filename)

        if os.path.exists(src_filepath):
            crop_and_resample(src_filepath, dst_filepath, geom, scale_factor)
        else:
            print("File not found:", src_filepath)
