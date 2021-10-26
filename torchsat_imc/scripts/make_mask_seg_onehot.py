"""
 * @author Lex Sherman
 * @email alexandershershakov@gmail.com
 * @create date 2021-08-27 11:00:00
 * @modify date 2021-08-27 11:00:00
 * @desc this tool is to patch the large satellite image to small image and label for segmentation.
"""

import os
import shapely
import argparse
import rasterio
import geopandas
import numpy as np
from pathlib import Path
from rasterio.windows import Window
from shapely.geometry import Polygon
from rasterio.features import rasterize

# ValueError: GEOSGeom_createLinearRing_r returned a NULL pointer. https://github.com/Toblerity/Shapely/issues/1005
import shapely
shapely.speedups.disable()

def split_image_and_label(  image_filepath: Path, label_dirpath: Path, label_classes: set, 
                            tile_size: int, drop_last: bool, image_outdir: Path, label_outdir: Path, id_separator: str, 
                            tile_ext: str, split_images: bool = True, split_labels: bool = True) -> bool:
    """ Rasterize vector geojson files into one raster image, 
        split image and rasterized label into tiles

        label_dir structure:
        .   
        ├── 'label_dir'
        │   ├── class1.geojson
        │   ├── class2.geojson
        │   ├── ...

        converts all geojson class files into one
        multichannel raster image and then splits it 

        Args:
            image_filepath (Path): full path to image file
            label_dir (Path): full path to label dir
            label_classes (set): list of classes in the label
            tile_size (int): tile size
            drop_last (bool): drop last tiles in the edges of the image
            image_outdir (Path): dir path for the output image tiles
            label_outdir (Path): dir path for the output label tiles
            id_separator (str): separator between tile id and tile description
            tile_ext (str): extension of all tiles (with dot, i.e.: '.tif')
    """

    class_count = len(label_classes)
    if class_count == 0:
        print("No label classes were specified!")
        return False

    for path in [image_outdir, label_outdir]:
        if not os.path.exists(path):
            os.makedirs(path)

    if not image_filepath.is_file():
        print(f"file {image_filepath} does not exits.")
        return False

    if not label_dirpath.is_dir():
        print(f"directory {label_dirpath} doesn't exist")
        return False

    # split image and label
    img_src = rasterio.open(image_filepath)
    rows = img_src.meta['height'] // tile_size if drop_last else img_src.meta['height'] // tile_size + 1
    cols = img_src.meta['width']  // tile_size if drop_last else img_src.meta['width']  // tile_size + 1
    for row in range(rows):
        for col in range(cols):
            try:
                window = Window(col * tile_size, row * tile_size, tile_size, tile_size)
                patched_transform = rasterio.windows.transform(window, img_src.transform)

                #
                # split image
                #

                if split_images:

                    patched_arr = img_src.read(window=window, boundless=True)
                    outfile_image = Path(image_outdir) / f"{image_filepath.stem}{id_separator}{row}_{col}{tile_ext}" # output image tile name
                    kwargs = img_src.meta.copy()
                    kwargs.update({
                        'height': window.height,
                        'width': window.width,
                        'transform': patched_transform})
                    with rasterio.open(outfile_image, 'w', **kwargs) as dst:
                        dst.write(patched_arr)

                #
                # split label
                #

                if split_labels:

                    outfile_label = Path(label_outdir) / f"{image_filepath.stem}{id_separator}{row}_{col}{tile_ext}" # output label tile name
                    bounds = rasterio.windows.bounds(window, img_src.transform) # clip geojson poligon
                    label_tile_arr = np.zeros((class_count, tile_size, tile_size), dtype=np.uint8)

                    for class_idx, class_filename in enumerate(os.listdir(label_dirpath)):
                        class_filepath = label_dirpath / class_filename
                        if not class_filepath.exists():
                            print(f"File {class_filepath} do not exist. Skipping this class")
                            continue

                        # read_file works with cyrillic filenames only with encoding='cp1251'
                        class_label_df = geopandas.read_file(class_filepath, encoding='cp1251') 
                        clipped_poly = geopandas.clip(class_label_df, Polygon.from_bounds(*bounds))

                        poly_shp = []
                        for geom in clipped_poly.geometry:
                            poly_shp.append((geom, 255))

                        if len(poly_shp) != 0:
                            label_tile_arr[class_idx, :,:] = rasterize(poly_shp, out_shape=(tile_size, tile_size), default_value=0, transform=patched_transform, dtype=np.uint8)   
                        else:
                            label_tile_arr[class_idx, :,:] = np.zeros((window.height, window.width), dtype=np.uint8)

                    # save multichannel onehot rasterized label
                    kwargs = img_src.meta.copy()
                    kwargs.update({
                        'driver': 'GTiff',    # Short format driver name (e.g. “GTiff” or “JPEG”)
                        'count': class_count, # Number of dataset bands
                        'height': window.height,
                        'width': window.width,
                        'transform': patched_transform, # Affine transformation mapping the pixel space to geographic space
                        'dtype': 'uint8'
                    })
                    with rasterio.open(outfile_label, 'w', **kwargs) as dst:
                        dst.write(label_tile_arr)

            # sometimes because of wrong shape of object, i.e. like this:
            #
            # |
            # |
            # |____
            # |   |
            # |   |
            # |___|
            #
            # Get TopologicalError
            #
            # TODO: show error to user
            except Exception as e:
                print(str(e))
                continue

    img_src.close()
    return True


def split_images_and_labels(item_ids: set, images_dirpath: Path, labels_dirpath: Path, classes: set, 
                            tile_size: int, drop_last: bool, image_outdir: Path, label_outdir: Path, id_separator: str, 
                            tile_ext: str, split_images: bool = True, split_labels: bool = True):
    """ Rasterize labels and save them as images

        Args:
            items_ids (set): item ids to rasterize
            images_dirpath (Path): path to the directory with images (features)
            labels_dirpath (Path): path to the directory with labels
            classes (set): list of label classes
            tile_size (int): size of tiles
            drop_last (bool): drop last tiles in the edges of the image
            image_outdir (Path): output directory for splitted images
            label_outdir (Path): output directory for splitted rasterized labels
            id_separator (str): separator between splitted tile id and description
            tile_ext (str): extension for tiles
    """

    image_filenames = os.listdir(images_dirpath)
    for id in item_ids:

        # find filename (with extension) that corresponds to the id
        found = False
        imagename = ""
        for filename in image_filenames:
            if Path(filename).stem == id:
                imagename = filename
                found = True

        if not found:
            continue
        
        # form params
        image_filepath = images_dirpath / imagename
        label_dirpath  = labels_dirpath / id

        split_image_and_label(image_filepath=image_filepath, 
                              label_dirpath=label_dirpath,
                              label_classes=classes,
                              tile_size=tile_size,
                              drop_last=drop_last,
                              image_outdir=image_outdir,
                              label_outdir=label_outdir,
                              id_separator=id_separator,
                              tile_ext=tile_ext,
                              split_images=split_images,
                              split_labels=split_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_filepath', type=str, help='full path to image file', required=True)
    parser.add_argument('--label_dir', type=str, help='full path to label dir', required=True)
    parser.add_argument('--label_classes', nargs='+', type=str, help='list of classes in the label dir', required=True)
    parser.add_argument('--tile_size', default=256, type=int, help='tile size of the patched image', required=True)
    parser.add_argument('--drop_last', default=True, type=bool, help='drop last tiles in the edges of the image', required=True)
    parser.add_argument('--image_outdir', type=str, help='dir path for the output image tiles', required=True)
    parser.add_argument('--label_outdir', type=str, help='dir path for the output label tiles', required=True)
    parser.add_argument('--id_separator', type=str, help='separator between tile id and tile description')
    parser.add_argument('--tile_ext', type=str, help='output tile extension (with dot, i.e.: .tif)')
    args = parser.parse_args()

    print(args)

    result = split_image_and_label( image_filepath = Path(args.image_filepath), 
                            label_dirpath = Path(args.label_dir), 
                            label_classes = set(args.label_classes), 
                            tile_size = args.tile_size, 
                            drop_last = args.drop_last, 
                            image_outdir = args.image_outdir, 
                            label_outdir = args.label_outdir,
                            id_separator = args.id_separator,
                            tile_ext = args.tile_ext) 
    if result == True:
        print("Successfull image and label splitting")
