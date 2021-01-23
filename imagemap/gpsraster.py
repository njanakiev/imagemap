import os
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.wkt
import shapely.geometry
import datashader as ds
import datashader.transfer_functions as tf
import rasterio as rio


def load_gps_subset(filepaths, extent, epsg=4326):
    extent_geometry = shapely.geometry.box(*extent)

    gdf_list = []
    for src_filepath in filepaths:
        print(src_filepath)

        df = pd.read_csv(src_filepath)
        df['geom'] = df.apply(
            lambda row: shapely.geometry.Point(row['lon'], row['lat']), 
            axis=1)
        # df.drop(columns=['lon', 'lat'], inplace=True)
        gdf = gpd.GeoDataFrame(df,
            geometry='geom',
            crs={'init': 'epsg:4326'})

        mask = gdf.within(extent_geometry)
        gdf = gdf[mask]

        print(gdf.shape)
        gdf_list.append(gdf)

    gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True),
        crs=gdf_list[0].crs)
    print(gdf.shape)

    return gdf


def create_geotiff(df, extent, dst_filepath, x='x', y='y', width=1000, height=1000, epsg=4326):
    cvs = ds.Canvas(plot_width=width, plot_height=height,
                    x_range=(extent[0], extent[2]),
                    y_range=(extent[1], extent[3]))
    
    agg = cvs.points(df, x, y)
    img = tf.shade(agg, cmap=['lightblue', 'darkblue'], how='log')

    X = img.data.copy()
    X = np.flip(X, axis=0)

    profile = {
        'driver': 'Gtiff',
        'dtype': 'uint32',
        'width': X.shape[1],
        'height': X.shape[0],
        'count': 1,
        'crs': rio.crs.CRS.from_epsg(epsg),
        'transform': rio.transform.from_bounds(
            *extent, X.shape[1], X.shape[0]),
        'nodata': 0
    }

    with rio.open(dst_filepath, 'w', **profile) as dst:
        dst.write(X, 1)
