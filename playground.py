import rasterio
# import rasterio.plot as rplt
# from rasterio.mask import mask
# from rasterio.transform import rowcol
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import os
# from pyproj import Transformer # required lib for facilatating coordinate transformation (between polygons and the raster)
import json
import shapely
# from shapely.ops import transform
from typing import List


rim_path = "Data/geotiffs/tier1/images" # images relative path
rpoly_path = "Data/geotiffs/tier1/labels" # labels relative path

## func design - load raster with polygons then send to the "chipped" into xy pairs ?? 


rimgs = os.listdir(rim_path) # list of all the raster images
rpolys = os.listdir(rpoly_path) # list of all the JSON files for the labels


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

    with open(rpoly_path + '/' + features_path, 'r') as f:
        data = json.load(f) # load the JSON file

    polys = [
        shapely.from_wkt(feature['wkt']) 
        for feature in data['features']['xy']  # list comprehension to iteratively retrive all the polygon coordinates in the file
    ]

    return polys




for idx, raster_path in enumerate(rimgs):

    #  namign convention - 
    # disaster-name_index_pre/post_disaster.tif
    # labels folder seems to follow the same convention - disaster-name_index_pre/post_disaster.json

    polygon_json = rpolys[idx]

    # sanity check to see if the correct poly has been selected
    
    if raster_path.split('.')[0] == polygon_json.split('.')[0]:

        # retrive polygon coordinates from JSON label
        polygons = get_poly(polygon_json)

        with rasterio.open(rim_path+'/'+raster_path) as raster:

            # print(raster.crs) --> ESPG:4326; need to ensure polygons/geometries follow the same CRS

            img = raster.read([1,2,3]) # returns 3-d array : i.e. (bands, h, w)

            # Raster IO → (bands, rows, cols)
            # plotting → (rows, cols, channels)
            img = img.transpose(1,2,0) # reorder the img array to follow (h,w,band) order as matplotlib adheres to such an order



            # coordinate transformer to ensure ploygons and raster follow the same CRS, preventing misalignment of polygons in the final plot
            # crs_trans = Transformer.from_crs(
            #     "ESPG:4326",
            #     raster.crs,
            #     always_xy=True
            # )
            
            # poly_proj = transform(crs_trans.transform, polygons)

            # plot raster
            fig, ax = plt.subplots(1,2,figsize=(10, 10))
            ax[0].set_title("Original raster")
            ax[0].imshow(img)

            transform = raster.transform

            for poly in polygons:
                # following piece of code to be used if lang_lat coords were use for polygns. however, there seems to be some offset in where the polys are drawn and where
                # they're supposed to be. need to look into this
                # coords = [(raster.index(x, y)[1], raster.index(x, y)[0])
                #           for x, y in zip(*poly.exterior.xy)]
                # patch = MplPolygon(coords, closed=True, facecolor='red', alpha=0.4)
                #ax.add_patch(patch)
                
                x, y = poly.exterior.xy

                # MplPolygon is used for drawing "patches", plt.plot will only draw line but this will allow more customization over poly as a patch over the image
                # requires a (x,y) pair tuples
                coords = list(zip(x, y))

                patch = MplPolygon(
                    coords,
                    closed=True,
                    facecolor="red",
                    edgecolor="red",
                    alpha=0.4
                )
                ax[1].set_title("Raster w Polys")
                ax[1].imshow(img)
                ax[1].add_patch(patch)

            plt.show()

        break


    else:

        assert f"Mismatched raster and polygon. \nRaster: {raster_path}\nPolygon: {polygon_json}"


