"""
 * @author sshuair
 * @email sshuair@gmail.com
 * @create date 2020-05-31 16:06:19
 * @modify date 2020-05-31 21:15:30
 * @desc this tool is to patch the large satellite image to small image and label for segmentation.
"""


import os
from glob import glob
from geopandas.geodataframe import GeoDataFrame
import numpy as np
from pathlib import Path
import geopandas

import shapely
from shapely.geometry import Polygon
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from tqdm import tqdm
import argparse

def rasterize_geojson(label_df: GeoDataFrame, img_src, width: int, height: int, drop_last: bool, label_outpath: str, image_file: str):
    """
    rasterizes and splits geojson
    """
    label_class_names = set(label_df['class_name'].values)
    class_count = len(label_class_names)
    label_tile_arr = np.zeros((class_count, width, height), dtype=np.uint8)
    
    rows = img_src.meta['height'] // height if drop_last else img_src.meta['height'] // height + 1
    cols = img_src.meta['width'] // width if drop_last else img_src.meta['width'] // width + 1
    for row in tqdm(range(rows)):
        for col in range(cols):
            # label
            outfile_label = Path(os.path.join(label_outpath, Path(image_file).stem + '_' + str(row) + '_' + str(col) + Path(image_file).suffix)).with_suffix('.tif')
            window = Window(col * width, row * height, width, height)
            patched_transform = rasterio.windows.transform(window, img_src.transform)

            # clip geojson poligon
            bounds = rasterio.windows.bounds(window, img_src.transform)
            clipped_poly = geopandas.clip(label_df, Polygon.from_bounds(*bounds))

            for i, label_class_name in enumerate(label_class_names):
                
                # get all geometries of the class
                shapes = []
                for geom, class_name in zip(clipped_poly.geometry, clipped_poly['class_name'].values):
                    if class_name == label_class_name:
                        shapes.append((geom, 255))
                
                # rasterize shapes
                if len(shapes) != 0:
                    label_tile_arr[i, :,:] = rasterize(shapes, out_shape=(width, height), default_value=0, transform=patched_transform, dtype=np.uint8)   
                else:
                    label_tile_arr[i, :,:] = np.zeros((height, width), dtype=np.uint8)
                
            # save multichannel onehot rasterized label
            kwargs = img_src.meta.copy()
            kwargs.update({
                'driver': 'GTiff', #  A short format driver name (e.g. “GTiff” or “JPEG”)
                'count': class_count, # The count of dataset bands
                'height': window.height,
                'width': window.width,
                'transform': patched_transform, # Affine transformation mapping the pixel space to geographic space
                'dtype': 'uint8'
            })
            with rasterio.open(outfile_label, 'w', **kwargs) as dst:
                dst.write(label_tile_arr)


def rasterize_shape(label_df: GeoDataFrame, img_src, width: int, height: int, drop_last: bool, label_outpath: str, image_file: str):
    """
    rasterizes and splits shp file
    """
    label_df['geometry'] = label_df.buffer(0) # fix of invalid shape exception

    label_class_names = set([label[0] for label in label_df.values])
    class_count = len(label_class_names)
    label_tile_arr = np.zeros((class_count, width, height), dtype=np.uint8)
    rows = img_src.meta['height'] // height if drop_last else img_src.meta['height'] // height + 1
    cols = img_src.meta['width'] // width if drop_last else img_src.meta['width'] // width + 1
    for row in tqdm(range(rows)):
        for col in range(cols):
            # label
            outfile_label = Path(os.path.join(label_outpath, Path(image_file).stem + '_' + str(row) + '_' + str(col) + Path(image_file).suffix)).with_suffix('.tif')
            window = Window(col * width, row * height, width, height)
            patched_transform = rasterio.windows.transform(window, img_src.transform)

            # clip shp poligon
            bounds = rasterio.windows.bounds(window, img_src.transform)
            clipped_poly = geopandas.clip(label_df, Polygon.from_bounds(*bounds))

            for i, label_class_name in enumerate(label_class_names):
                
                # get all geometries of the class
                shapes = []
                for geom, shape_list in zip(clipped_poly.geometry, clipped_poly.values): # shape_list is [class_name, POLYGON]
                    if shape_list[0] == label_class_name:
                        shapes.append((geom, 255))

                # rasterize shapes
                if len(shapes) != 0:
                    label_tile_arr[i, :,:] = rasterize(shapes, out_shape=(width, height), default_value=0, transform=patched_transform, dtype=np.uint8)   
                else:
                    label_tile_arr[i, :,:] = np.zeros((height, width), dtype=np.uint8)
                
            # save multichannel onehot rasterized label
            kwargs = img_src.meta.copy()
            kwargs.update({
                'driver': 'GTiff', #  A short format driver name (e.g. “GTiff” or “JPEG”)
                'count': class_count, # The count of dataset bands
                'height': window.height,
                'width': window.width,
                'transform': patched_transform, # Affine transformation mapping the pixel space to geographic space
                'dtype': 'uint8'
            })
            with rasterio.open(outfile_label, 'w', **kwargs) as dst:
                dst.write(label_tile_arr)

def make_mask_seg(image_file: str, label_file: str, width: int, height: int, drop_last: bool, image_outpath: str, label_outpath: str):

    for path in [image_outpath, label_outpath]:
        if not os.path.exists(path):
            os.makedirs(path)

    for file in [image_file, label_file]:
        if not Path(file).is_file():
            raise ValueError('file {} does not exits.'.format(file))

    # read label and image
    label_df = geopandas.read_file(label_file)
    img_src = rasterio.open(image_file)

    # split image
    rows = img_src.meta['height'] // height if drop_last else img_src.meta['height'] // height + 1
    cols = img_src.meta['width'] // width if drop_last else img_src.meta['width'] // width + 1
    for row in tqdm(range(rows)):
        for col in range(cols):
            # image
            outfile_image = os.path.join(image_outpath, Path(image_file).stem+'_'+str(row)+'_'+str(col)+Path(image_file).suffix)
            window = Window(col * width, row * height, width, height)
            patched_arr = img_src.read(window=window, boundless=True)
            kwargs = img_src.meta.copy()
            patched_transform = rasterio.windows.transform(window, img_src.transform)
            kwargs.update({
                'height': window.height,
                'width': window.width,
                'transform': patched_transform})
            with rasterio.open(outfile_image, 'w', **kwargs) as dst:
                dst.write(patched_arr)
    
    # split rasterized vector
    if label_file.endswith(".geojson"):
        rasterize_geojson(label_df, img_src, width, height, drop_last, label_outpath, image_file)
    elif label_file.endswith(".shp"):
        rasterize_shape(label_df, img_src, width, height, drop_last, label_outpath, image_file)
    else: 
        raise Exception("Not supported vector file format")
    
    img_src.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', type=str, help='the target satellite image to split. Note: the file should have crs')
    parser.add_argument('--label_file', type=str, help='''the corresponding label file of the satellite image. 
                vector or raster file. Note the crs should be same as satellite image.''')
    parser.add_argument('--width', default=256, type=int, help='the width of the patched image')
    parser.add_argument('--height', default=256, type=int, help='the height of the patched image')
    parser.add_argument('--drop_last', default=True, type=bool,
              help='set to True to drop the last column and row, if the image size is not divisible by the height and width.')
    parser.add_argument('--image_outpath', type=str, help='image output dir path')
    parser.add_argument('--label_outpath', type=str, help='label output dir path')
    args = parser.parse_args()

    make_mask_seg(args.image_file, args.label_file, args.width, args.height, args.drop_last, image_outpath=args.image_outpath, label_outpath=args.label_outpath)




# def generate_mask(raster_path, shape_df):
#     """
#     Function that generates a binary mask from a vector file (shp or geojson)
#     https://lpsmlgeo.github.io/2019-09-22-binary_mask/
    
#     raster_path = path to the .tif;
#     shape_path = path to the shapefile or GeoJson.
#     output_path = Path to save the binary mask.
#     file_name = Name of the file.
#     """
#     with rasterio.open(raster_path, "r") as src:
#         raster_img = src.read()
#         raster_meta = src.meta

#     print("CRS Raster: {}, CRS Vector {}".format(shape_df.crs, src.crs))

#     #Generate polygon
#     def poly_from_utm(polygon, transform):
#         poly_pts = []
        
#         poly = shapely.ops.cascaded_union(polygon)
#         for i in np.array(poly.exterior.coords):
            
#             # Convert polygons to the image CRS
#             poly_pts.append(~transform * tuple(i))
            
#         # Generate a polygon object
#         new_poly = Polygon(poly_pts)
#         return new_poly

#     # Generate Binary maks

#     poly_shp = []
#     im_size = (src.meta['height'], src.meta['width'])
#     for num, row in shape_df.iterrows():
#         if row['geometry'].geom_type == 'Polygon':
#             poly = poly_from_utm(row['geometry'], src.meta['transform'])
#             poly_shp.append(poly)
#         else:
#             for p in row['geometry']:
#                 poly = poly_from_utm(p, src.meta['transform'])
#                 poly_shp.append(poly)

#     print("poly_shp: ", poly_shp)

#     mask = rasterize(shapes=poly_shp,
#                      out_shape=im_size, )

#     return mask
