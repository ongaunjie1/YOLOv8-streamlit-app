from ultralytics import YOLO
import streamlit as st
import cv2
from pytube import YouTube
import tempfile
import easyocr 
import numpy as np

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    
    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)
    
    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_youtube_video(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    
    source_youtube = st.sidebar.text_input("YouTube Video url")

    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(0)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_frame,
                                                     image
                                                     )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")

def upload_easyocr(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            detected_frames(conf,
                                            model,
                                            st_frame,
                                            image)
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")

def detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    reader = easyocr.Reader(['en'], gpu=False)

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Create a copy of the original frame to modify
    modified_frame = image.copy()

    for i, license_plate in enumerate(res[0].boxes.data.tolist()):
        x1, y1, x2, y2, score, class_id = license_plate

        # Process license plate
        license_plate_image_gray = cv2.cvtColor(modified_frame[int(y1):int(y2), int(x1):int(x2), :], cv2.COLOR_BGR2GRAY)
        _, license_plate_image_thresh = cv2.threshold(license_plate_image_gray, 64, 255, cv2.THRESH_BINARY_INV)

        # Read license plate image
        detections = reader.readtext(license_plate_image_thresh)

        if detections:
            detected_plate_text = detections[0][1]  # Extract the detected text

            # Replace the class name with the detected license plate text
            license_plate_text = detected_plate_text

            # Draw a bounding box around the detected car plate
            cv2.rectangle(modified_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Calculate the text size
            text_size, _ = cv2.getTextSize(license_plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

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
            cv2.rectangle(modified_frame, (background_x1, background_y1), (background_x2, background_y2), background_color, -1)

            # Draw the centered text on the enlarged filled background
            text_color = (255, 255, 255)  # Text color
            cv2.putText(modified_frame, license_plate_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    # Display the modified frame with updated bounding boxes and license plate text
    st_frame.image(modified_frame, caption='Detected Video', channels="BGR", use_column_width=True)

# Function to apply flood fill and other processing to the image
def process_license_plate(license_plate_image, floodfill_threshold, threshold_block_size, brightness):
    # Convert to grayscale
    license_plate_image_gray = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)

    # Replace dark values (below floodfill_threshold) with floodfill_threshold
    license_plate_image_gray[license_plate_image_gray < floodfill_threshold] = floodfill_threshold

    # Adjust brightness
    license_plate_image_bright = cv2.convertScaleAbs(license_plate_image_gray, alpha=1, beta=brightness)

    # Apply adaptive thresholding to create a black-and-white image
    license_plate_image_thresh = cv2.adaptiveThreshold(license_plate_image_bright, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, threshold_block_size, 2)

    return license_plate_image_thresh

    
