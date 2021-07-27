
pip install .\wheels\geopandas-0.9.0-py3-none-any.whl
pip install .\wheels\rasterio-1.1.8-cp36-cp36m-win_amd64.whl
pip install -r .\requirements.txt

## Rasterize geojson and save it as tif

python torchsat\cli\make_mask_seg.py --image_file D:/Projects/Github/torchsat/torchsat/scripts/18.tif --label_file  D:/Projects/Github/torchsat/torchsat/scripts/18.geojson --width 128 --height 128 --field notused --outpath D:\Projects\Github\torchsat\torchsat\scripts\out\label

## Split raster image into tiles

python torchsat\cli\make_mask_cls.py --filepath D:/Projects/Github/torchsat/torchsat/scripts/18.tif --width 128 --height 128  --outpath D:\Projects\Github\torchsat\torchsat\scripts\out\feature
