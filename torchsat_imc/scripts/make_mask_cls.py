"""
 * @author Lex Sherman
 * @email alexandershershakov@gmail.com
 * @create date 2021-08-27 11:00:00
 * @modify date 2021-08-27 11:00:00
 * @desc this tool is to patch the large satellite image to small image
"""

import os
import click
import rasterio
import argparse
from tqdm import tqdm
from glob import glob
from pathlib import Path
from rasterio.windows import Window

color_map = {
    0: (0, 0, 0, 255),
    1: (31, 119, 180, 255),
    2: (174, 199, 232, 255),
    3: (255, 127, 14, 255),
    4: (255, 187, 120, 255),
    5: (44, 160, 44, 255),
    6: (152, 223, 138, 255),
    7: (214, 39, 40, 255),
    8: (255, 152, 150, 255),
    9: (148, 103, 189, 255),
    10: (197, 176, 213, 255),
    11: (140, 86, 75, 255),
    12:(196, 156, 148, 255),
    13: (227, 119, 194, 255),
    14: (247, 182, 210, 255),
    15: (127, 127, 127, 255),
    16: (199, 199, 199, 255),
    17: (188, 189, 34, 255),
    18: (219, 219, 141, 255),
    19: (23, 190, 207, 255),
    20: (158, 218, 229)
}

def split_image(filepath: Path, tile_size: int, drop_last: bool, outpath: Path, stem_separator: str, tile_ext: str, colormap: bool):
    """ Split single image into tiles. 
        Each tile is saved to the output directory and has a name in a following format:
        {filename.stem}{stem_separator}_{row}_{col}{tile_ext}

        Args:
            filepath (Path): full path to an image
            tile_size (int): size of a tile
            drop_last (bool): drop last tiles in the edges of the image
            outpath (Path): dir path for the output tiles
            stem_separator (str): separator between tile id and tile description
            tile_ext (str): extension of a tile
            colormap (bool): generate color interpretation for each band in the image
    """
    with rasterio.open(filepath, 'r') as src:
        rows = src.meta['height'] // tile_size if drop_last else src.meta['height'] // tile_size + 1
        columns = src.meta['width'] // tile_size if drop_last else src.meta['width'] // tile_size + 1
        for row in tqdm(range(rows)):
            for col in range(columns):
                outfile = os.path.join(outpath, f"{filepath.stem}{stem_separator}_{row}_{col}{tile_ext}")
                window = Window(col * tile_size, row * tile_size, tile_size, tile_size)
                patched_arr = src.read(window=window, boundless=True)
                kwargs = src.meta.copy()
                kwargs.update({
                    'height': window.height,
                    'width': window.width,
                    'transform': rasterio.windows.transform(window, src.transform)})
                with rasterio.open(outfile, 'w', **kwargs) as dst:
                    dst.write(patched_arr)
                    if colormap:
                        dst.write_colormap(1, color_map)

def make_mask_cls(image_names: set, input_path: Path, tile_size: int, drop_last: bool, outpath: Path, stem_separator: str, tile_ext: str, colormap: bool):
    """ Split all images in the directory on tiles
    
        Args:
            image_names (set): names of all images to split on tile with extensions
            input_path (Path): path to the directory with images to split
            tile_size (int): tile size
            drop_last (bool): drop last tiles in the edges of the image
            outpath (Path): path to the output directory for tiles
            stem_separator (str): separator between tile id and tile description
            tile_ext (str): extension of tiles
            colormap (bool): generate color interpretation for each band in the image
    """
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    for image_name in image_names:
        split_image(filepath = input_path / image_name, 
                    tile_size = tile_size,
                    drop_last = drop_last, 
                    outpath = outpath, 
                    stem_separator = stem_separator,
                    tile_ext = tile_ext,
                    colormap = colormap)
