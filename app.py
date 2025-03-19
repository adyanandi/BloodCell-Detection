import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Path to your locally saved model (in the Downloads folder)
model_path = os.path.join(os.path.dirname(__file__), 'best.pt') # Use your local path


model = YOLO(model_path)

# Helper function for detection (removed confidence threshold parameter)
def detect_image(image):
    image = np.array(image.convert('RGB'))
    results = model(image)

    # Annotate image
    annotated_img = results[0].plot()

    boxes = results[0].boxes
    class_names = model.names
    detections = []

    class_count = {class_names[i]: 0 for i in range(len(class_names))}

    for box in boxes:
        cls = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        if conf >= 0.5:  # Default threshold of 0.5
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            class_name = class_names[cls]
            class_count[class_name] += 1
            detections.append({
                "Class": class_name,
                "Confidence": round(conf, 2),
                "BoundingBox": str(xyxy)
            })

    return annotated_img, detections, class_count
 
# Title
st.title("🔬 Blood Cell Detection Web App")


# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'upload'  # Default page

# Function to navigate between pages
def switch_page(page_name):
    st.session_state.page = page_name

# Upload Image Page
if st.session_state.page == 'upload':
    st.write("Upload a microscope image to detect RBC, WBC, and Platelets")
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("🔍 Run Detection"):
        # Store uploaded file in session state and switch to detection page
        st.session_state.uploaded_file = uploaded_file
        switch_page('detection')

# Detection Result Page
elif st.session_state.page == 'detection':
    st.subheader("🧑‍🔬 Detection Output")
    uploaded_file = st.session_state.get('uploaded_file', None)
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Perform detection without confidence threshold
        annotated_img, detections, class_count = detect_image(image)

        # Display the annotated image with bounding boxes
        st.image(annotated_img, caption="Detection Output", use_column_width=True)
        st.subheader("📊 Detected Object Counts")
        # Create a two-column layout for pie chart and bar chart
        col1, col2 = st.columns(2)
        

        with col1:
            # Display object count chart (pie chart)
            st.subheader("Pie Chart")
            fig, ax = plt.subplots()
            ax.pie(class_count.values(), labels=class_count.keys(), autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)

        with col2:
            # Display object count bar chart
            st.subheader("Bar Chart")
            count_df = pd.DataFrame(list(class_count.items()), columns=["Class", "Count"])
            st.bar_chart(count_df.set_index('Class')['Count'])

        # Display Detected Object Counts in Sidebar
        st.sidebar.subheader("📋 Detection Table")
        st.sidebar.dataframe(pd.DataFrame(detections))

        # Display Performance Metrics
        st.subheader("📈 Model Performance (Precision & Recall)")
        metrics_data = {
            "Class": ["RBC", "WBC", "Platelets", "Overall"],
            "Precision": [0.91, 0.85, 0.88, 0.88],
            "Recall": [0.89, 0.83, 0.86, 0.86]
        }
        df_metrics = pd.DataFrame(metrics_data)
        st.table(df_metrics)

        # Display Object Counts Centrally in Sidebar
        st.sidebar.subheader("📊 Object Counts")
        object_count_html = "<div style='display: flex; justify-content: center; gap: 10px;'>"
        for cell_type, count in class_count.items():
            object_count_html += f"<div style='font-size: 16px;'><b>{cell_type}:</b> {count}</div>"
        object_count_html += "</div>"

        # Render object counts HTML in the sidebar
        st.sidebar.markdown(object_count_html, unsafe_allow_html=True)

    # Option to go back to the upload page
    if st.button("Back to Upload Page"):
        switch_page('upload')








