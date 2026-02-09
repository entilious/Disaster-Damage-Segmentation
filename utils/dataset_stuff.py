# Utility funcs for concoting da dataset

import os
import shapely
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List
from numpy.typing import NDArray
import json
import cv2

rasters_dir = "Data/geotiffs/tier1/images"
labels_dir = "Data/geotiffs/tier1/labels"


def create_dirs():
    '''
    Docstring for create_dirs

    create directories for train-test split
    '''
    cwd = os.getcwd()
    os.makedirs(os.path.join(cwd, 'model_data/train/img'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'model_data/train/mask'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'model_data/test/img'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'model_data/test/mask'), exist_ok=True)

    print(f"Directories created or already exist at {cwd+'model_data'}"); return



def get_poly(features_path: str) -> List:
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


def extract_mask(raster_image: NDArray[np.uint8], poly_coords : List) -> NDArray[np.bool_]:
    '''
    Docstring for extract_mask
    
    :param raster_image: 3-d np.array containing pixel info of the raster image
    :type raster_image: NDArray[np.uint8]
    :return: 1-d array containing a binary mask of the image
    :rtype: NDArray[bool_]
    '''
    height, width = raster_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for poly in poly_coords:
        # Extracts polygon boundary points -> ensures image bounds are adhered to -> fills them on the mask
        poly_pts = np.array(poly.exterior.coords, dtype=np.int32)
        poly_pts[:, 0] = np.clip(poly_pts[:, 0], 0, width - 1) # ensurin' polygons DO NOT exceed image bounds
        poly_pts[:, 1] = np.clip(poly_pts[:, 1], 0, height - 1)
    
        cv2.fillPoly(mask, [poly_pts], 255)

    return mask


rasters = os.listdir(rasters_dir)
lables = os.listdir(labels_dir)

for idx, raster_path in enumerate(rasters):

    poly_path = lables[idx]

    if raster_path.split('.')[0] == poly_path.split('.')[0]:

        polys = get_poly(poly_path)

        with rasterio.open(rasters_dir + '/' + raster_path) as r:

            img = r.read([1,2,3]) # returns 3-d array : i.e. (bands, h, w)

            # Raster IO → (bands, rows, cols)
            # plotting/ml stuff → (rows, cols, channels/bands)
            img = img.transpose(1,2,0) # transpose the img array to follow (h,w,band) order

            mask = extract_mask(raster_image=img, poly_coords=polys)
            
            fig, ax = plt.subplots(1,2,figsize=(10, 10))
            ax[0].imshow(img)
            ax[1].imshow(mask, cmap='gray')
            plt.show()

        print("we out")
        break

    else:
        assert f"Mismatched paths: Image and Label paths are not the same. Raster path: {raster_path}\nLabel path: {poly_path}"

        

