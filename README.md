# YOLO-Satellite-Detection

## Overview
This project involves the training and evaluation of object detection models using YOLOv7, YOLOv9, and improved versions for the classification of satellite imagery, specifically for vehicle detection in the SIMD dataset. The following steps are undertaken:

1. **Training YOLOv7 Models**: Utilized the official YOLOv7 implementation and improved versions with advanced model configurations to train for satellite imagery classification.
2. **Training YOLOv9 Model**: Trained a YOLOv9 model independently and further enhanced it by integrating modules from the improved YOLOv7 models.
3. **Training YOLOv5 and YOLOv8**: Adapted improved versions of YOLOv5 to YOLOv8, leveraging new backbone components from YOLOv5 modules.

## Dataset
The **Satellite Imagery Multi-vehicles Dataset (SIMD)** is used for training and evaluation. The dataset includes satellite images labeled with vehicle types, enabling effective classification. 

## Training Steps

### Part 1: Training YOLOv7
1. **Clone YOLOv7 repository**:
   ```bash
   git clone https://github.com/WongKinYiu/yolov7.git
   cd yolov7
   ```

2. **Train YOLOv7 (Base Model)**:
   Train YOLOv7 on SIMD dataset for 40 to 50 epochs:
   ```bash
   python train.py --img 640 --batch 16 --epochs 50 --data SIMD.yaml --weights yolov7.pt
   ```

3. **Train Improved YOLOv7 Models**:
   Train three different improved versions of YOLOv7:
   - `yolov7-transBoT3-ConvNext-attention.yaml`
   - `yolov7-C3C2-CA.yaml`
   - `yolov7-Swin-HorNet.yaml`

   Example:
   ```bash
   python train.py --cfg yolov7-transBoT3-ConvNext-attention.yaml --epochs 50 --data SIMD.yaml
   ```

4. **Compare Results**:
   Compare the results of all trained models using the metrics available in the `run/train/` folder.

### Part 2: Training YOLOv9
1. **Clone YOLOv9 repository**:
   ```bash
   git clone https://github.com/WongKinYiu/yolov9.git
   cd yolov9
   ```

2. **Train YOLOv9**:
   Train YOLOv9 independently on the SIMD dataset:
   ```bash
   python train.py --img 640 --batch 16 --epochs 50 --data SIMD.yaml --weights yolov9.pt
   ```

3. **Integrate Modules from Improved YOLOv7 Models**:
   Add two or three updated modules from the improved YOLOv7 models to YOLOv9 and train the new model.

### Part 3: Training YOLOv5 and YOLOv8
1. **Clone YOLOv5 and YOLOv8 repositories**:
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   git clone https://github.com/ultralytics/ultralytics.git
   ```

2. **Train YOLOv5 and Improved YOLOv5**:
   - Train the `EMO/yolov5-EMO-CSRMBC.yaml` configuration on the SIMD dataset.

3. **Integrate Improved YOLOv5 Modules into YOLOv8**:
   - Update YOLOv8 with modules from improved YOLOv5 and train the model.

## Evaluation
- The trained models are evaluated based on detection accuracy, precision, recall, and other metrics to determine the most efficient configuration for satellite image vehicle detection.
- Comparison metrics are stored and analyzed in the `run/train/` directory.
- part 1: (first model)
  ![image](https://github.com/user-attachments/assets/1354769c-63a3-4994-962b-3fb60ee5612b)

  Run the detect.py file:
  ```bash
   !python detect.py --weights 
   /content/drive/MyDrive/yolov7/runs/train/exp53/weights/best.pt --source 
   /content/drive/MyDrive/yolov7/sat-1/test/images
   ```
  ![image](https://github.com/user-attachments/assets/3127a9a6-40bd-42bf-bfa2-0bee1983f286)

  Run the test.py file:
  ```bash
   !python test.py --data /content/drive/MyDrive/yolov7/sat-1/data.yaml -
   batch 16 --conf 0.001 --iou 0.65 --img 640 --device 0 --weights 
   /content/drive/MyDrive/yolov7/runs/train/exp53/weights/best.pt 
   ```
  ![image](https://github.com/user-attachments/assets/97dc9f86-bf99-4d52-bf44-49cf95feb52b)

  an example:
  ![image](https://github.com/user-attachments/assets/0fb7d1b5-9eb9-4bba-9b2d-94ef141db6b8)



## links
- **YOLOv7**: https://github.com/WongKinYiu/yolov7
- **YOLOv9**: https://github.com/WongKinYiu/yolov9
- **YOLOv5 and YOLOv8**: https://github.com/ultralytics/ultralytics - https://github.com/ultralytics/yolov5.git
- **dataset**: https://drive.google.com/drive/folders/1-EinPidvqA9rjyLuPaxvJTtmKEI2JkAr?usp=drive_link


