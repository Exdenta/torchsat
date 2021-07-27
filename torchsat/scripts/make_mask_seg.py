"""
 * @author sshuair
 * @email sshuair@gmail.com
 * @create date 2020-05-31 16:06:19
 * @modify date 2020-05-31 21:15:30
 * @desc this tool is to patch the large satellite image to small image and label for segmentation.
"""


import os
from glob import glob
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

def generate_mask(raster_path, shape_df):
    """
    Function that generates a binary mask from a vector file (shp or geojson)
    https://lpsmlgeo.github.io/2019-09-22-binary_mask/
    
    raster_path = path to the .tif;
    shape_path = path to the shapefile or GeoJson.
    output_path = Path to save the binary mask.
    file_name = Name of the file.
    """
    with rasterio.open(raster_path, "r") as src:
        raster_img = src.read()
        raster_meta = src.meta

    print("CRS Raster: {}, CRS Vector {}".format(shape_df.crs, src.crs))

    #Generate polygon
    def poly_from_utm(polygon, transform):
        poly_pts = []
        
        poly = shapely.ops.cascaded_union(polygon)
        for i in np.array(poly.exterior.coords):
            
            # Convert polygons to the image CRS
            poly_pts.append(~transform * tuple(i))
            
        # Generate a polygon object
        new_poly = Polygon(poly_pts)
        return new_poly

    # Generate Binary maks

    poly_shp = []
    im_size = (src.meta['height'], src.meta['width'])
    for num, row in shape_df.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            poly = poly_from_utm(row['geometry'], src.meta['transform'])
            poly_shp.append(poly)
        else:
            for p in row['geometry']:
                poly = poly_from_utm(p, src.meta['transform'])
                poly_shp.append(poly)

    print("poly_shp: ", poly_shp)

    mask = rasterize(shapes=poly_shp,
                     out_shape=im_size, )

    return mask


def make_mask_seg(image_file: str, label_file: str, field, width: int, height: int, drop_last: bool, outpath: str):

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    if not Path(image_file).is_file():
        raise ValueError('file {} not exits.'.format(image_file))

    # read the file and distinguish the label_file is raster or vector
    try:
        label_src = rasterio.open(label_file)
        label_flag = 'raster'
    except rasterio.RasterioIOError:
        label_df = geopandas.read_file(label_file)
        label_flag = 'vector'

    img_src = rasterio.open(image_file)
    rows = img_src.meta['height'] // height if drop_last else img_src.meta['height'] // height + 1
    columns = img_src.meta['width'] // width if drop_last else img_src.meta['width'] // width + 1
    for row in tqdm(range(rows)):
        for col in range(columns):
            # image
            outfile_image = os.path.join(outpath, Path(image_file).stem+'_'+str(row)+'_'+str(col)+Path(image_file).suffix)
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

            # label
            outfile_label = Path(outfile_image).with_suffix('.tif')
            if label_flag == 'raster':
                label_arr = label_src.read(window=window, boundless=True)
            else:
                bounds = rasterio.windows.bounds(window, img_src.transform)
                clipped_poly = geopandas.clip(label_df, Polygon.from_bounds(*bounds))
                shapes = [(geom, 255) for geom in clipped_poly.geometry]
                # shapes = [(geom, value) for geom, value in zip(clipped_poly.geometry, clipped_poly[field])]
                if len(shapes) != 0:
                    label_arr = rasterize(shapes, out_shape=(width, height), default_value=0, transform=patched_transform)   
                else:
                    label_arr = np.zeros((window.height, window.width), dtype=np.uint8)

            kwargs = img_src.meta.copy()
            kwargs.update({
                'driver': 'GTiff',
                'count': 1,
                'height': window.height,
                'width': window.width,
                'transform': patched_transform,
                'dtype': 'uint8'
            })
            with rasterio.open(outfile_label, 'w', **kwargs) as dst:
                dst.write(label_arr, 1)

    img_src.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', type=str, help='the target satellite image to split. Note: the file should have crs')
    parser.add_argument('--label_file', type=str, help='''the corresponding label file of the satellite image. 
                vector or raster file. Note the crs should be same as satellite image.''')
    parser.add_argument('--field', type=str, help='field to burn')
    parser.add_argument('--width', default=256, type=int, help='the width of the patched image')
    parser.add_argument('--height', default=256, type=int, help='the height of the patched image')
    parser.add_argument('--drop_last', default=True, type=bool,
              help='set to True to drop the last column and row, if the image size is not divisible by the height and width.')
    parser.add_argument('--outpath', type=str, help='the output file path')
    args = parser.parse_args()

    make_mask_seg(args.image_file, args.label_file, args.field, args.width, args.height, args.drop_last, args.outpath)
