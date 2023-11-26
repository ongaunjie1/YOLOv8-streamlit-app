# yolov8-streamlit-app 
* An object/segmentation detection app created using streamlit. Refer below for all the features of the app
* Model used: https://github.com/ultralytics/ultralytics
* App link: https://yolov8-app.streamlit.app/
* Link to EasyOCR: https://github.com/JaidedAI/EasyOCR
* NOTE: The streamlit app might not work if the GitLFS's free bandwidth exceeds 1 gb.

# App Features 
* Object **detection/segmentation** using pre-trained yoloV8 model (trained on Open Images V7 dataset with 600 distinct classes) , refer to openimages.txt for the list of objects detectable using the base model
  
* Custom-trained yolov8 model **for detecting potholes** 
  
* Custom-trained yolov8 model **for detecting car license plates** 
  
* Integrated license plate detector with EasyOCR for **reading license plates**. Additionally, added an image preprocessing function to handle images with brightness and Image glare issues
  
* Custom-trained yolov8 model **to detect PPE** (7 classes: ['Protective Helm', 'Shield', 'Jacket', 'Dust Mask', 'Eye Wear, 'Glove, 'Protective Boots')
  
* To use your custom trained model, just add your .pt files into the weights and make some minor changes to the settings.py and app.py files **(Note: If your model's weights are >25mb, you will need to us Git LFS to upload your files)**

# Custom-trained model's result:
|    Custom-trained models    |      mAP50      | mAP50-95|
|---------------              |-----------------|-------  |
| Potholes detection (yolov8m)          |       0.721     | 0.407   |
| Car License plate detection (yolov8m)|       0.995     | 0.828   |
| PPE Detection  (yolov8m)              |       0.991     | 0.738   | 

* mAP50 and mAP50-95 are metrics used to evaluate object detection models.
* mAP50 measures the average precision at an Intersection over Union (IoU) threshold of 0.5, while mAP50-95 considers the average precision across IoU thresholds from 0.5 to 0.95.
* Higher values indicate better accuracy and robustness in detecting objects across different IoU levels.
* IOU is the ratio of the area of overlap between the predicted and actual bounding boxes to the area of their union

# What is EasyOCR ?
* EasyOCR is an open-source Python library for Optical Character Recognition (OCR). Its primary purpose is to recognize and extract text from images, enabling applications to convert images containing printed or handwritten text into machine-readable and editable text.

# How was it used in this project ?

1) The fine-tuned yolov8 model is used for the license plate detection in an image, accurately locating the license plate's position.

2) The detected license plate region is cropped from the original image to isolate the license plate.

3) The cropped license plate image is converted to black and white, simplifying the image and emphasizing text features.

4) OpenCV (cv2) is used to enhance the contrast of the black and white license plate image, making the text more distinct and improving OCR accuracy.

5) EasyOCR is applied to the preprocessed license plate image to perform Optical Character Recognition (OCR), extracting and reading the text content from the license plate.

# How to use the app?
## Main page of the app
![image](https://github.com/ongaunjie1/yolov8-streamlit/assets/118142884/cb74c49e-a5dc-499d-ba20-a61a38b1919f)

## Sidebar of the app
![image](https://github.com/ongaunjie1/yolov8-streamlit/assets/118142884/6a181ef5-1a5c-4168-ba5f-e615273edfd9)
* **Step 1**: Select a task (Detection, segmentation, Potholes detection etc.)
* **Step 2**: Adjust model confidence if needed
* **Step 3**: Select source type (image/video)
* **Step 4**: Upload the image/video

## Example usage: License plate detection 
![image](https://github.com/ongaunjie1/yolov8-streamlit/assets/118142884/ac717b2b-84b0-4d15-a137-ad1a27e9db96)
* After uploading an image, click on the Detect objects button on the sidebar

### After prediction (image on the right)
![image](https://github.com/ongaunjie1/yolov8-streamlit/assets/118142884/68334ace-0051-4f75-9f1e-3ed7b5d1d4f8)

## Example 2: Reading License plate using EasyOCR
![image](https://github.com/ongaunjie1/yolov8-streamlit/assets/118142884/f72dde77-135d-4559-891d-65c2e16ad764)

# Possible improvements for the app
* Improve the accuracy of the custom-trained models (train on higher quality data, Data augmentation, Hyperparameter tuning(computationally intensive, time-consuming)
* Integrate car license plate detector with **SORT/DeepSORT** which keep tracks of the car's information. (For real-world use case)
* Experiment with using **different size** yolov8 models (smaller models offers faster inference but less accuracy), smaller size models may be more suitable if you're deploying your app on Streamlit's Community Cloud

# Issues
* Currently webcam feature isn't working after deploying to streamlit cloud but it works locally.
* App is currently deployed at Streamlit's Community Cloud which has limited resource, which may crash the app if the resources are exceeded.
* Video processing are slow running on Streamlit Cloud (Deploying the app on a paid-cloud service would help with the processing speed)

# Steps for fine-tuning a yolov8 model: 
## Example: Car license plate detection

## Step 1: Annotate your custom images using RoboFlow
![image](https://github.com/ongaunjie1/yolov8-streamlit/assets/118142884/ee93751a-922a-466f-bc9e-a90392cfda2f)

## Step 2: Split images into train, val, test
![image](https://github.com/ongaunjie1/YOLOv8-streamlit-app/assets/118142884/279223ed-48da-44e4-876f-ac611522451c)

## Step 3: Pre-process the images (Resize, change orientation and etc)
![image](https://github.com/ongaunjie1/YOLOv8-streamlit-app/assets/118142884/bee4e263-d10a-4856-9f7e-baea2be024a6)

## Step 4: Augment the images if needed
![image](https://github.com/ongaunjie1/yolov8-streamlit/assets/118142884/7e81982b-38a1-4373-abdb-d93ff51c766c)

## Step 5: Select the appropriate yolov8 model 
![image](https://github.com/ongaunjie1/yolov8-streamlit/assets/118142884/5f10edbf-e12b-4588-89f6-5cb4d14fe2bc)

## Step 6: Fine-tune the yolov8 model using the annotated images (fine-tune using personal gpu or use a gpu from google colab)
### RoboFlow will generate a .yaml file automatically, verify it if needed and then proceed with training the model
![image](https://github.com/ongaunjie1/yolov8-streamlit/assets/118142884/3e02d882-73ca-4ef8-b1ce-4251212f9f6f)
```
!yolo task=detect mode=train model=yolov8m.pt data=/content/drive/MyDrive/carplate/car-plate-detection-1/data.yaml epochs=70 imgsz=640 
```

## Step 7: Validate fine-tuned model
```
!yolo task=detect mode=val model=/content/drive/MyDrive/carplate/runs/detect/train/weights/best.pt data=/content/drive/MyDrive/carplate/car-plate-detection-1/data.yaml
```
![image](https://github.com/ongaunjie1/yolov8-streamlit/assets/118142884/242109a8-3859-455c-b0b3-65f2d2c7ccb7)

## Step 8: Inference the custom model on test dataset
```
!yolo task=detect mode=predict model=/content/drive/MyDrive/carplate/runs/detect/train/weights/best.pt conf=0.25 source=/content/drive/MyDrive/carplate/car-plate-detection-1/test/images
```
### Verify any one of the images
![image](https://github.com/ongaunjie1/yolov8-streamlit/assets/118142884/c673fc62-7dbe-4b94-8d39-241d56cf4522)

## NOTE: The example above trained the model using yolov8's default hyperparmeters. Feel free to change some of the values of the hyperparameters to see if it improves the accuracy of the model.



  

