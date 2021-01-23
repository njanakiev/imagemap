import numpy as np
import scipy.linalg
import scipy.spatial
from PIL import Image
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


def get_square_bounds(bounds):
    x0, y0, x1, y1 = bounds
    w, h = (x1 - x0), (y1 - y0)

    if w > h:
        cy = (y0 + y1) * 0.5
        return [x0, cy - 0.5 * w, x1, cy + 0.5 * w]
    else:
        cx = (x0 + x1) * 0.5
        return [cx - 0.5 * h, y0, cx + 0.5 * h, y1]

def quad_rectangle_bounds(bounds):
    x0, y0, x1, y1 = bounds
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5

    return ([x0, y0, cx, cy],
            [cx, y0, x1, cy],
            [x0, cy, cx, y1],
            [cx, cy, x1, y1])


def scale_bounds(bounds, w, h):
    x0, y0, x1, y1 = bounds
    W, H = (x1 - x0), (y1 - y0)
    
    if (W / H) > (w / h):
        cy = (y0 + y1) * 0.5
        factor = 0.5 * ((W * h) / w)
        return [x0, cy - factor, x1, cy + factor]
    else:
        cx = (x0 + x1) * 0.5
        factor = 0.5 * ((H * w) / h)
        return [cx - factor, y0, cx + factor, y1]


def get_bounds(bounds, w, h, boundary_type='outer'):
    x0, y0, x1, y1 = bounds
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


def normalize_aspect(X):
    X_out = X.copy()
    min_x, min_y = X.min(axis=0) 
    max_x, max_y = X.max(axis=0)
    w, h = max_x - min_x, max_y - min_y
    
    if w > h:
        X_out[:, 0] = (X_out[:, 0] - min_x) / w
        X_out[:, 1] = (X_out[:, 1] - min_y) / w + (w - h) / (2 * w)
    else:
        X_out[:, 0] = (X_out[:, 0] - min_x) / h + (h - w) / (2 * h)
        X_out[:, 1] = (X_out[:, 1] - min_y) / h
    
    return X_out


def calc_dispersion(X, min_r, step_size=0.2, noise=None):
    D = np.zeros(X.shape)
    kdtree = scipy.spatial.cKDTree(X)
    num_neighbors = np.zeros((X.shape[0],))
    
    for curr_idx in range(X.shape[0]):
        idx_list = kdtree.query_ball_point(
            X[curr_idx], min_r)
        num_neighbors[curr_idx] = len(idx_list)
        
        if noise is not None:
            D[curr_idx] += np.random.normal(0, noise, size=(X.shape[1],))
        
        for idx in idx_list:
            dist = scipy.linalg.norm(X[idx] - X[curr_idx])
            if dist > 0.0:
                D[curr_idx] -= step_size * (min_r - dist) * \
                    ((X[curr_idx] - X[idx]) / dist)
    
    return D, num_neighbors


def disperse_points(X, min_r, step_size, noise=None, iterations=200, verbose=False):
    X_out = X.copy()
    
    for i in range(iterations):
        D, num_neighbors = calc_dispersion(X_out, min_r, step_size, noise)
        if verbose and i % 20 == 0:
            print(f"Iteration: {i: 5d}, "\
                  f"Max num neighbors: {num_neighbors.max()}")
        
        X_out = X_out - D
    
    return X_out
