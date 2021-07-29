
pip install .\wheels\geopandas-0.9.0-py3-none-any.whl
pip install .\wheels\rasterio-1.1.8-cp36-cp36m-win_amd64.whl
pip install -r .\requirements.txt

# Train image segmentation model

## 1. Rasterize geojson and save it as tif

python scripts\make_mask_seg_onehot.py --image_file D:/Projects/torchsat/scripts/18.tif --label_file  D:/Projects/torchsat/scripts/18.geojson --width 128 --height 128 --image_outpath D:\Projects\torchsat\datasets\temp\train\image --label_outpath D:\Projects\torchsat\datasets\temp\train\label

## 2. Split raster image into tiles

python torchsat\cli\make_mask_cls.py --filepath D:/Projects/Github/torchsat/torchsat/scripts/18.tif --width 128 --height 128  --outpath D:\Projects\Github\torchsat\datasets\train\image
