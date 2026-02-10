# Utility funcs for concoting da dataset

import shapely
import rasterio
import numpy as np
from typing import List
from numpy.typing import NDArray
import json
import cv2


def get_poly(labels_dir : str, features_path: str) -> List:
    '''
    Docstring for get_poly
    
    Helper func to retrieve polygon coordinate(s) from the associated JSON labels file

    :param features_path: Path to the label JSON file 
    :type features_path: str
    '''

    # the polygons are expressed as WKTs (Well-Known-Text) in the JSON, so we using shapely instead of fiona as fiona expects GeoJSON not WKT
        # JSON {
        #       features: {
        #           lang_lat: [{
        #               ....,
        #               wkt: {...},  ----> interested in this
        #                       },
        #                       {..},
        #                       ...],
        #           xy: [{
        #               ....,
        #               wkt: {...},  ----> interested in this. bc i'm segmentation models does not need CRS and allat math, we using pixel coords rather than lang_lat
        #                       },
        #                       {..},
        #                       ...]
        #           }
        #}

    with open(labels_dir + '/' + features_path, 'r') as f:
        data = json.load(f) # load the JSON file

    polys = [
        shapely.from_wkt(feature['wkt']) 
        for feature in data['features']['xy']  # list comprehension to iteratively retrive all the polygon coordinates in the file
    ]

    return polys


def extract_mask(raster_image: NDArray[np.uint8], labels_dir : str , labels_path : str) -> NDArray[np.bool_]:
    '''
    Docstring for extract_mask
    
    :param raster_image: 3-d np.array containing pixel info of the raster image
    :type raster_image: NDArray[np.uint8]
    :return: 1-d array containing a binary mask of the image
    :rtype: NDArray[bool_]
    '''

    poly_coords = get_poly(labels_dir=labels_dir, features_path=labels_path)

    height, width = raster_image.shape[1:]
    mask = np.zeros((height, width), dtype=np.uint8)

    for poly in poly_coords:
        # Extracts polygon boundary points -> ensures image bounds are adhered to -> fills them on the mask
        poly_pts = np.array(poly.exterior.coords, dtype=np.int32)
        poly_pts[:, 0] = np.clip(poly_pts[:, 0], 0, width - 1) # ensurin' polygons DO NOT exceed image bounds
        poly_pts[:, 1] = np.clip(poly_pts[:, 1], 0, height - 1)
    
        cv2.fillPoly(mask, [poly_pts], 255)

    return mask


def get_raster_data(rasters_dir : str, f_path : str) -> NDArray:

    '''
    Docstring for get_raster_data
    
    :param rasters_dir: string containinf the directory path of the images
    :type rasters_dir: str
    :param f_path: string containing the directory path for the image in question
    :type f_path: str
    :return: array containing image data in H,W,C format
    :rtype: NDArray
    '''

    with rasterio.open(rasters_dir + '/' + f_path) as r:

        img = r.read([1,2,3]) # returns 3-d array : i.e. (bands, h, w)

        # Raster IO → (bands, rows, cols)
        # plotting/ml stuff → (rows, cols, channels/bands)
        # since no plottin being done and torch tensors also expect c,h,w we wont transponse now
        # img = img.transpose(1,2,0) # transpose the img array to follow (h,w,band) order

    return img


        

