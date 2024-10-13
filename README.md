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

### Results

The project is capable of detecting various objects in images, segmenting them, recognizing their colors, and analyzing their spatial relationships. Each step in the process is visualized, making it easy to interpret the model's performance.

---

