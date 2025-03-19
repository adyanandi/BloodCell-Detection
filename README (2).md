
# Blood Cell Detection Web App

This web application utilizes state-of-the-art object detection (YOLOv8) to identify and count different types of blood cells (RBC, WBC, Platelets) from uploaded microscope images. It provides users with visual output, such as annotated images with bounding boxes, pie charts showing the distribution of blood cells, and detailed tables summarizing detection results. This app is useful for medical professionals or researchers who need quick and accurate detection and analysis of blood cells in images.


## Demo

![Blood Cell Detection Demo](https://i.imgur.com/tyTlEN3.gif)



## Features

- **Image Upload & Detection**: Upload a microscope image and automatically detect blood cells like RBC, WBC, and Platelets using YOLOv10.
- **Real-Time Object Detection**: Visualize the detection results in real-time with bounding boxes and labels for each detected cell type.
- **Interactive Pie Chart**: View a pie chart of the detected object counts, providing a visual breakdown of the detected blood cells.
- **Detection Table**: A table displays detailed information about each detected object, including the class, confidence, and bounding box coordinates.
- **Model Performance Metrics**: The app displays performance metrics (Precision & Recall) for each detected object type (RBC, WBC, Platelets).
- **Responsive Layout**: The app provides a two-column layout where the detection output image, pie chart, and bar chart are shown side by side for an optimized and interactive user experience.
- **Cross-Platform Support**: The web app is designed to be accessible on various platforms and devices, ensuring easy access to users across different environments.



## Tech Stack

**Frontend:**  
- **Streamlit**: Used for building the interactive web application with a simple user interface for uploading images and displaying results.
- **Pandas**: Utilized for managing and displaying detection results and performance metrics in tabular form.
- **Matplotlib**: Used for creating pie charts and visualizing object detection statistics.
- **PIL (Pillow)**: For image processing, including loading and displaying images.

**Backend:**
- **Python**: The primary language for the backend, performing blood cell detection using YOLOv8.
- **Ultralytics YOLO**: A pre-trained model used for object detection in microscope images to identify and classify blood cells.
- **PyTorch**: Framework used for running the YOLOv8 model.

**Other Tools:**
- **NumPy**: Used for numerical operations during image processing.

- **Matplotlib**: Used for generating charts and graphs, including pie charts and bar charts for displaying detection statistics.



## Installation

To run this Blood Cell Detection Web App locally, follow the steps below:

1. Clone the repository:

```bash
  git clone https://github.com/yourusername/blood-cell-detection.git
   cd blood-cell-detection
```
2.Create a virtual environment (optional but recommended):
```bash
pip install pipenv
```
```bash
pipenv shell
```
```bash
pipenv install
```
3. Install Requirements
```bash
pip install -r requirements.txt
```
4. Model (best.pt): The best.pt model file is already included in the repository, so you do not need to download it separately. It will be available when you clone the repo

5. Run the Streamlit App
```bash 
streamlit run app.py
```
    