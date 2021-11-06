import numpy as np
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS


def get_exif_metadata(filepath):
    im = Image.open(filepath)

    exif = im._getexif()
    item = {}
    if exif is None:
        return item
    
    for (tag, value) in exif.items():
        decoded = TAGS.get(tag, tag)
        if decoded == "GPSInfo":
            gps_data = {}
            for t in value:
                sub_decoded = GPSTAGS.get(t, t)
                gps_data[sub_decoded] = value[t]

            item[decoded] = gps_data
        else:
            item[decoded] = value

    return item


def _convert_to_degress(value):
    """Helper function to convert the GPS coordinates stored in the EXIF to degress in float format"""
    #d = float(value[0][0]) / float(value[0][1])
    #m = float(value[1][0]) / float(value[1][1])
    #s = float(value[2][0]) / float(value[2][1])
    d, m, s = [float(v) for v in value]
    return d + (m / 60.0) + (s / 3600.0)


def get_lat_lon(gps_info):
    """Returns the latitude and longitude, from the provided exif_data"""
    lat, lon = None, None

    if isinstance(gps_info, dict):
        gps_latitude = gps_info.get("GPSLatitude")
        gps_latitude_ref = gps_info.get('GPSLatitudeRef')
        gps_longitude = gps_info.get('GPSLongitude')
        gps_longitude_ref = gps_info.get('GPSLongitudeRef')
        
        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = _convert_to_degress(gps_latitude)
            if gps_latitude_ref != "N":
                lat = 0 - lat

            lon = _convert_to_degress(gps_longitude)
            if gps_longitude_ref != "E":
                lon = 0 - lon

    return lat, lon


def square_extent(extent):
    # TODO: boundary_type inner or outer
    x0, y0, x1, y1 = extent
    w, h = (x1 - x0), (y1 - y0)

    if w > h:
        cy = (y0 + y1) * 0.5
        return [x0, cy - 0.5 * w, x1, cy + 0.5 * w]
    else:
        cx = (x0 + x1) * 0.5
        return [cx - 0.5 * h, y0, cx + 0.5 * h, y1]


def quad_rectangle_extent(extent):
    x0, y0, x1, y1 = extent
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5

    return ([x0, y0, cx, cy],
            [cx, y0, x1, cy],
            [x0, cy, cx, y1],
            [cx, cy, x1, y1])


def scale_extent(extent, w, h, boundary_type='outer'):
    x0, y0, x1, y1 = extent
    W, H = (x1 - x0), (y1 - y0)
    
    if ((W / H) < (w / h) and boundary_type == 'inner') or \
       ((W / H) > (w / h) and boundary_type == 'outer'):
        cy = (y0 + y1) * 0.5
        factor = 0.5 * ((W * h) / w)
        return [x0, cy - factor, x1, cy + factor]
    else:
        cx = (x0 + x1) * 0.5
        factor = 0.5 * ((H * w) / h)
        return [cx - factor, y0, cx + factor, y1]


def relative_extent(src_extent, dst_extent, size, invert_y=False):
    w, h = size
    xa0, ya0, xa1, ya1 = src_extent
    xb0, yb0, xb1, yb1 = dst_extent
    
    dpx = (xa1 - xa0) / w
    dpy = (ya1 - ya0) / h
    
    x0 = int((xb0 - xa0) / dpx)
    x1 = int((xb1 - xa0) / dpx)
    y0 = int((yb0 - ya0) / dpy)
    y1 = int((yb1 - ya0) / dpy)
    
    if invert_y:
        y1 = h - y0
        y0 = h - y1
        
    rel_extent = [x0, y0, x1, y1]
    rel_size = (x1 - x0, y1 - y0)
    
    return rel_extent, rel_size


def normalize_aspect(coords):
    coords_out = coords.copy()
    min_x, min_y = coords_out.min(axis=0) 
    max_x, max_y = coords_out.max(axis=0)
    w = max_x - min_x
    h = max_y - min_y
    
    if w > h:
        coords_out[:, 0] = (coords_out[:, 0] \
                         - min_x) / w
        coords_out[:, 1] = (coords_out[:, 1] \
                         - min_y) / w + (w - h) / (2 * w)
    else:
        coords_out[:, 0] = (coords_out[:, 0] \
                         - min_x) / h + (h - w) / (2 * h)
        coords_out[:, 1] = (coords_out[:, 1] \
                         - min_y) / h
    
    return coords_out


def image_grid(
    images, 
    nrow, ncol,
    tile_size=129,
    padding=0,
    image_paths=False
):
    w = ncol * (tile_size + padding) + padding
    h = nrow * (tile_size + padding) + padding
    grid_image = Image.new(
        'RGB', (w, h), (255, 255, 255))

    for idx in range(nrow*ncol):
        if idx < len(images):
            img = Image.open(images[idx]) if image_paths else images[idx]
            img_square = ImageOps.fit(img, (tile_size, tile_size))
            img_square = img_square.resize(
                (tile_size, tile_size))
            
            i = idx % ncol
            j = (idx - i) // ncol
            offset = (
                (tile_size + padding) * i + padding,
                (tile_size + padding) * j + padding
            )
            grid_image.paste(img_square, offset)

    return grid_image
