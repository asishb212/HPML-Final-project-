# Object Detection In Thermal Images Through YOLO Models 
## Thermal image dataset [click here](https://www.flir.com/oem/adas/adas-dataset-form/)

## Objective

The objective of this project is to develop a fast and lightweight model for object detection in thermal images using YOLO (You Only Look Once) models. The model is trained using transfer learning on YOLO backbones and tested on the FLIR dataset, which contains grayscale images of data collected from heat-maps generated by IR cameras mounted on cars driving on streets. The goal is to provide a deployable model that can detect objects from IR sensory data and be used in various applications such as cars, drones, traffic cameras, and dash-cams.

## Challenges
The project faces several technical challenges, including loss of features in thermal images, computationally expensive training, testing, and benchmarking, and building a custom YOLO model that is optimized for the dataset. To address these challenges, the project team used PyTorch Framework on a Linux platform, with a custom convolution model based on scaling down larger YOLO models to attain an optimal tradeoff between accuracy and parameters.

## Parallelism
The project team used distributed data parallelism to cut down training time for the custom lightweight model, and SGD Optimizer with momentum and a constant batch size of 16. They also tested pruning to check if the loss in accuracy is worth the speedup, and min-max normalized the input images. They benchmarked the custom YOLO model under different hyper-parameters against different models to test for optimal parameters.

## Benchmarking
The project team ran multiple variants of different models to benchmark for optimal parameters and trained across multiple GPUs and different parallelisms for different models. They tested and trained the models on NYU HPC (GPU RTX800, 6 CPUs, 32GB Memory, (GPU A100 for testing)). They also tested deployment on video input data, with performance up to 24fps expected on Google Colab.

## Results
The custom YOLO model built achieved accuracy close to larger 5l and 5x models but had a number of parameters and TTA/convergence time comparable to that of the 5s model. The custom model has comparable accuracy to the larger YOLO models but uses a fraction of the parameters and layers and has a faster inference time. The project team's technical contribution includes creating a custom YOLO model optimized for the FLIR dataset and testing it on different hyper-parameters and different models to find optimal parameters. The custom model is lightweight, making it ideal for novel applications such as drones.


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
