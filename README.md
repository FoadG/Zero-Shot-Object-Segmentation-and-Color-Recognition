# Zero-Shot Object Segmentation and Color Recognition

This project demonstrates a workflow for object segmentation and color recognition using zero-shot learning techniques, combining various deep learning models to achieve state-of-the-art results. 

### Overview

In this project, we implemented a system that can detect, segment, and recognize the colors of various objects using the following steps:

1. **Object Detection (Zero-Shot Learning)**:
   We use the YOLO-World model for zero-shot object detection. YOLO-World allows detection of objects based on textual prompts without requiring prior training on specific categories. This is done by specifying class names of interest directly through prompts, such as "car" or "person".

2. **Segmentation**:
   After detecting the objects, the **Segment Anything Model (SAM)** is applied to precisely segment the detected objects. SAM generates accurate masks for each detected object based on its bounding box.

3. **Color Recognition**:
   For each segmented object, a Convolutional Neural Network (CNN) model is used to classify the dominant color. The CNN has been trained to recognize 15 different colors:
   - Beige
   - Black
   - Blue
   - Brown
   - Gold
   - Green
   - Grey
   - Orange
   - Pink
   - Purple
   - Red
   - Silver
   - Tan
   - White
   - Yellow

4. **Relative Distance Measurement**:
   In addition to segmentation and color recognition, the project includes functionality to measure the relative distance between detected objects within the image. Objects that are close to each other are identified, enhancing spatial understanding in a scene.

### Key Features

- **Zero-Shot Detection**: No need to pre-train the model for specific object categories. YOLO-World can infer the desired objects based on textual prompts.
- **Highly Accurate Segmentation**: SAM provides high-quality segmentation masks for detected objects, ensuring precision in complex scenes.
- **Color Recognition**: A custom-trained CNN model accurately predicts the color of each segmented object from a list of 15 colors.
- **Distance Estimation**: Measure and compare the relative distances between detected objects, identifying which objects are close to each other.

### Technologies and Tools

- **YOLO-World**: Used for zero-shot object detection.
- **Segment Anything Model (SAM)**: Utilized for object segmentation.
- **Custom CNN**: Trained to classify 15 colors for the segmented objects.
- **Python**: The primary language for development.
- **PyTorch**: Used for deep learning tasks such as training and deploying the CNN model.
- **OpenCV**: For image processing and visualization.

### Setup and Installation

1. Clone the YOLO-World repository:
   ```bash
   git clone https://github.com/AILab-CVC/YOLO-World.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r YOLO-World/requirements/basic_requirements.txt
   pip install -r YOLO-World/requirements/demo_requirements.txt
   pip install -r YOLO-World/requirements/onnx_requirements.txt
   ```

3. Install additional dependencies:
   ```bash
   pip install opencv-python==4.9.0.80
   pip install opencv-python-headless
   pip install supervision
   pip install onnx onnxruntime onnxsim
   pip install mmcv==2.0.0 mmdet==3.0.0 mmengine==0.10.3 mmyolo==0.6.0
   pip install timm==0.6.13 transformers==4.36.2 albumentations
   ...
   ```

4. Load and prepare the model for inference using YOLO-World, segment using SAM, and classify colors using the custom CNN model.

### Results

The project is capable of detecting various objects in images, segmenting them, recognizing their colors, and analyzing their spatial relationships. Each step in the process is visualized, making it easy to interpret the model's performance.

---

