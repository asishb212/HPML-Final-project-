# Object Detection In Thermal Images Through YOLO Models 

## Thermal image dataset [click here](https://www.flir.com/oem/adas/adas-dataset-form/)

## Install dependencies
'''
pip install -qr yolov5/requirements.txt
'''

## Training
'''
python train.py --img 416 --batch 16 --epochs 200 --data /content/yolov5/data.yaml --cfg /content/yolov5/models/yolo5s.yaml --weights '' --name yolov5s_results  --cache
'''

## Inference
'''
%mkdir /yolov5/content/run
%cd /yolov5/content/yolov5/
!python detect.py --weights /content/best.pt --img 416 --conf 0.4 --source /content/part/
'''
