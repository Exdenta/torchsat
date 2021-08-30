
pip install .\wheels\geopandas-0.9.0-py3-none-any.whl
pip install .\wheels\rasterio-1.1.8-cp36-cp36m-win_amd64.whl
pip install -r .\requirements.txt

# Train image segmentation model

## 1. Rasterize geojson and save it as tif

python scripts\make_mask_seg_onehot.py --image_file D:/Projects/torchsat/scripts/18.tif --label_file  D:/Projects/torchsat/scripts/18.geojson --width 128 --height 128 --image_outpath D:\Projects\torchsat\datasets\temp\train\image --label_outpath D:\Projects\torchsat\datasets\temp\train\label

## 2. Split raster image into tiles

python torchsat\cli\make_mask_cls.py --filepath D:/Projects/Github/torchsat/torchsat/scripts/18.tif --width 128 --height 128  --outpath D:\Projects\Github\torchsat\datasets\train\image

## 3. Train

python scripts\train_seg.py --train-path=D:\Projects\torchsat\datasets\temp\train --val-path=D:\Projects\torchsat\datasets\temp\val --mean 0.34412036 0.36172597 0.32255994 --std 0.15378592 0.13528091 0.14353549 --image_extensions tif --label_extension tif --model unet34 --pretrained False --num-classes 2 --in-channels 3 --crop-size 128 --device cpu --batch-size 4 --epochs 90 --lr 0.001 --print-freq 1 --ckp-dir D:\Projects\torchsat\checkpoints

