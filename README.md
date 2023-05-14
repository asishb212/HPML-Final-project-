# Object Detection In Thermal Images Through YOLO Models 
## Thermal image dataset [click here](https://www.flir.com/oem/adas/adas-dataset-form/)

# Repository structure
```
HPML-Final-project-
├── final.ipynb
└── yolov5
    ├── benchmarks.py
    ├── classify
    ├── data
    ├── detect.py
    ├── models
    │   ├── yolov5l.yaml
    │   ├── yolov5m.yaml
    │   ├── yolov5n.yaml
    │   ├── yolov5s.yaml
    │   └── yolov5x.yaml
    ├── requirements.txt
    ├── runs
    │   └── train
    │       ├── singleGPU
    │       │   ├── yolov5l_results2
    │       │   │   ├── metrics
    │       │   │   └── weights
    │       │   │       ├── best.pt
    │       │   │       └── last.pt
    │       │   └── Other single GPU runs (custom , n,s,x)
    │       ├── yolov5l_dist_results2
    │       │   └── metrics
    │       │   └── weights
    │       │       ├── best.pt
    │       │       └── last.pt
    │       ├── All other multi GPU runs (custom , n,s,x)
    ├── segment
    │   ├── helper functions
    ├── train.py
    ├── utils
    │   ├── Helper functions
    └── val.py
```
# commands to execute the code
## Install dependencies
```
pip install -qr yolov5/requirements.txt
```
## Training for 5s backbone
<p> yaml file required </p>
```
train: path to train dataset
val: path to validation dataset
```
<p></p>
```
python yolov5/train.py --img 416 --batch 16 --epochs 200 --data yolov5/data.yaml --cfg yolov5/models/yolo5s.yaml --weights '' --name yolov5s_results  --cache
```
## Inference
```
!python yolov5/detect.py --weights /content/best.pt --img 416 --conf 0.4 --source /content/part/
```
