<h1><p align="center">Vehicles Counting using YOLOv4 + DeepSORT + Flask + Ngrok</p>
<p align="center">🚙 🛵 🚛 🚌</p>
</h1>
<p align="center">A project for counting vehicles using <code>YOLOv4</code> for training, <code>DeepSORT</code> for tracking, <code>Flask</code> for deploying to web (watch result purpose only) and <code>Ngrok</code> for public IP address </p>
<p align="center"><img src="./data/images/result.gif"/></p>

## Getting Started
This project has 3 main parts:
1. [Preparing data](#preparing-data)
2. [Training model using the power of YOLOv4](#training-model-using-yolov4)
3. [Implementing DeepSORT algorithm for counting vehicles](#implementing-deepsort-algorithm-for-counting-vehicles)

### Shortcuts
Note: For private reason, please ***ask for permission*** before using datasets and pre-trained model!
|Shortcuts|Links|
|:--:|:--:|
|📕 Colab notebooks|[Part 1](https://colab.research.google.com/drive/1Iur7UE3i2fV3Ka3Zw3Owqq2Y2d1MIhCE?usp=sharing), [Part 2](https://colab.research.google.com/drive/1Q75vbva305OQ8Dg60WpJwpXjwFO_qwWA?usp=sharing), [Part 3](https://colab.research.google.com/drive/1uTWscUDaqieHrNtg9puUuQqgs1w5WFtW?usp=sharing)|
|📀 Datasets|[Daytime](https://drive.google.com/file/d/1nCo9WVlxucc8C_tpropgvuIAZZ1A9ta9/view?usp=sharing), [Nighttime](https://drive.google.com/file/d/1s8OaXDya2tDjTKl3h3AF6r4uGOSymmNA/view?usp=sharing)|
|🚂 My pre-trained model|[GGDrive Mirror](https://drive.google.com/file/d/1-0lo7naWZUhTzJ94Yn4flSG7PSGR3ZZn/view?usp=sharing) (Works well in well-lit conditions)|



## Preparing data
[![Preparing data notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Iur7UE3i2fV3Ka3Zw3Owqq2Y2d1MIhCE?usp=sharing)

I splitted my data into 2 scenes: `daytime` and `nighttime`, and training 8 classes (4 classes each scene, which are `motorbike, car, bus, truck`).

Prepare your own data or **you can download my cleaned data with annotations**:
- Daytime dataset: [GGDrive mirror](https://drive.google.com/file/d/1nCo9WVlxucc8C_tpropgvuIAZZ1A9ta9/view?usp=sharing)
- Nighttime dataset: [GGDrive mirror](https://drive.google.com/file/d/1s8OaXDya2tDjTKl3h3AF6r4uGOSymmNA/view?usp=sharing)

If you prepare your own data, remember your annotation files fit this format:

1. Every image has its own annotation file (`.txt`)
2. Each file contains a list of objects' bounding box ([read this for more details](https://github.com/AlexeyAB/Yolo_mark/issues/60#issuecomment-401854885)):
  ```
  <object-id> <x> <y> <width> <height>
  ...
  ```
## Training model using YOLOv4
[![Training model notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q75vbva305OQ8Dg60WpJwpXjwFO_qwWA?usp=sharing)

>Training model on your local computer is really complicated in environment installation and slow-like-a-snail if you don't have a powerful GPU. In this case, I used **Google Colab**.

Read more: [Testing your trained model on local machine with OpenCV](./utils_obj_detection)

## Implementing DeepSORT algorithm for counting vehicles
[![Implementing DeepSORT notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uTWscUDaqieHrNtg9puUuQqgs1w5WFtW?usp=sharing)

First, setting up environment on your machine:
### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip
(TensorFlow 2 packages require a pip version > 19.0.)
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt

# Google Colab
!pip install -r requirements-colab.txt
```

### Convert YOLOv4 model to Tensorflow Keras
Copy your trained model in previous part to this project and run `save_model.py` in cmd:

- `--weights`: Path to `.weights` file (your trained model)
- `--output`: Path to converted model.
- `--model`: Model version (`yolov4` in this case) 

```bash
python save_model.py --weights ./yolov4_final.weights --output ./checkpoints/yolov4-416 --model yolov4
```

>Download my `.weights` model if you want: [GGDrive mirror](https://drive.google.com/file/d/1-0lo7naWZUhTzJ94Yn4flSG7PSGR3ZZn/view?usp=sharing)

### Counting now!
Import `VehiclesCounting` class in `object_tracker.py` file and using `run()` to start running:
```python
# Import this main file
from object_tracker import VehiclesCounting

# Initialize
# check the list of parameters below to modify values as you want
# check object_tracker.py file to check the default values

vc = VehiclesCounting()

# Run it
vc.run()
```

`VehicleCounting`'s parameters:
- `file_counter_log_name`: input your file counter log name
- `framework`: choose your model framework (tf, tflite, trt)
- `weights`: path to your .weights
- `size`: resize images to
- `tiny`: (yolo,yolo-tiny)
- `model`: (yolov3,yolov4)
- `video`: path to your video or set 0 for webcam or youtube url
- `output`: path to your results
- `output_format`: codec used in VideoWriter when saving video to file
- `iou`: iou threshold
- `score`: score threshold
- `dont_show`: dont show video output
- `info`: show detailed info of tracked objects
- `detect_line_position`: (0..1) of height of video frame.
- `detect_line_angle`: (0..180) degrees of detect line.

## Contact me
- [Facebook](https://www.facebook.com/duonggg.ne/)
- [LinkedIn](https://www.linkedin.com/in/duonggg/)
- [Email](mailto:duong.jt.19@gmail.com)

## References
I want to give my big thanks to all of these authors' repo:
- [yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort)
- [
Multi-Camera-Live-Object-Tracking](https://github.com/LeonLok/Multi-Camera-Live-Object-Tracking)
