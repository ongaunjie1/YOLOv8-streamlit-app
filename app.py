# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st
import easyocr  
reader = easyocr.Reader(['en'], gpu=False) 
import cv2
import numpy as np

# Local Modules
import settings
import helper


# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection using YOLOv8")

# Sidebar header
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation', 'Potholes Detection', 'License Plate Detection' , 'License Plate Detection with EasyOCR', 'PPE Detection'])

# Sidebar slider
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100
if model_type == 'License Plate Detection with EasyOCR':
    # Slider for floodfill threshold
    floodfill_threshold = st.sidebar.slider('Floodfill Threshold', 0, 250, 100, step=1)

    # Slider for thresholding block size
    threshold_block_size = st.sidebar.slider('Threshold Block Size (odd number, > 1)', 3, 201, 101, step=2)

    # Slider for brightness adjustment
    brightness = st.sidebar.slider('Brightness Adjustment', -100, 100, 0, step=1)

# Selecting Detection Or Segmentation
@st.cache_resource
def get_model_path(model_type):
    if model_type == 'Detection':
        return Path(settings.DETECTION_MODEL)
    elif model_type == 'Segmentation':
        return Path(settings.SEGMENTATION_MODEL)
    elif model_type == 'Potholes Detection':
        return Path(settings.CUSTOM_MODEL1)
    elif model_type == 'License Plate Detection':
        return Path(settings.CUSTOM_MODEL2)
    elif model_type == 'License Plate Detection with EasyOCR':
        return Path(settings.CUSTOM_MODEL2)
    elif model_type == 'PPE Detection':
        return Path(settings.CUSTOM_MODEL3)

# Load Pre-trained ML Model
@st.cache_resource
def load_model(model_path):
    try:
        model = helper.load_model(model_path)
        return model
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

model_path = get_model_path(model_type)
model = load_model(model_path)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                if len(boxes) == 0:
                    st.error("No objects detected in the image.")
                else:
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image',
                            use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")

        if source_img is not None and model_type == 'License Plate Detection with EasyOCR':
            if st.sidebar.button('Read License Plate'):
            # Read and display the uploaded image
                original_image = PIL.Image.open(source_img)
                original_image = np.array(original_image)

                # Detect license plates
                license_plates = model.predict(original_image)

                if not len(license_plates[0].boxes) == 0:
                    for i, license_plate in enumerate(license_plates[0].boxes):
                        x1, y1, x2, y2 = license_plate.xyxy[0]

                        # Crop license plate from the original image
                        license_plate_image = original_image[int(y1):int(y2), int(x1):int(x2)]

                        # Process the license plate image with adjustable parameters
                        processed_license_plate = helper.process_license_plate(license_plate_image, floodfill_threshold, threshold_block_size, brightness)

                    #Perform OCR on the processed license plate image
                    detections = reader.readtext(processed_license_plate)
                    if detections:
                            detected_plate_text = detections[0][1]  # Extract the detected text

                            # Replace the class name with the detected license plate text
                            license_plate_text = detected_plate_text

                            # Draw a bounding box around the detected car plate
                            cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                            # Calculate the text size
                            text_size, _ = cv2.getTextSize(license_plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 2)

                            # Calculate the position to center the text
                            text_x = int(x1 + (x2 - x1 - text_size[0]) / 2)
                            text_y = int(y1 - 10)

                            # Calculate the background size (adjust as needed)
                            background_width = text_size[0] + 20  # Add some extra space for padding
                            background_height = text_size[1] + 10  # Add some extra space for padding

                            # Calculate the position for the background
                            background_x1 = text_x - 10
                            background_y1 = text_y - text_size[1] - 5  # Adjusted to be higher
                            background_x2 = background_x1 + background_width
                            background_y2 = text_y + 5  # Adjusted to be lower

                            # Draw the enlarged filled background for the text
                            background_color = (0, 0, 0)
                            cv2.rectangle(original_image, (background_x1, background_y1), (background_x2, background_y2), background_color, -1)

                            # Draw the centered text on the enlarged filled background
                            text_color = (255, 255, 255)  # Text color
                            cv2.putText(original_image, license_plate_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, text_color, 2)
               
                    # Convert the modified image back to PIL format for displaying with Streamlit
                    image_with_text = PIL.Image.fromarray(original_image)

                    # Display the modified image with updated bounding boxes and license plate text
                    st.image(image_with_text, caption='Detected License Plate',
                            use_column_width=True)  
                    # Display the processed license plate image
                    st.image(processed_license_plate, caption='Processed License Plate', use_column_width=True)
                else:
                  st.error("No license plates detected in the image.")
          
elif source_radio == settings.VIDEO:
    if model_type == 'License Plate Detection with EasyOCR':
        helper.upload_easyocr(confidence, model)
    else:
        helper.infer_uploaded_video(confidence, model)

elif source_radio == settings.WEBCAM:
    if model_type in ['Detection', 'Segmentation', 'Potholes Detection', 'License Plate Detection', 'PPE Detection']:
        helper.play_webcam(confidence, model)
    else:st.error("Webcam is not supported for License Plate Detection with EasyOCR")

elif source_radio == settings.YOUTUBE:
    if model_type in ['Detection', 'Segmentation', 'Potholes Detection', 'License Plate Detection', 'PPE Detection']:
        helper.play_youtube_video(confidence, model)
    else:st.error("Youtube is not supported for License Plate Detection with EasyOCR")

else:
    st.error("Please select a valid source type!")
