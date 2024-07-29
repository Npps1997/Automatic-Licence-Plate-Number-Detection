# Automatic License Plate Detection

## Project Overview

This project focuses on real-time detection of license plates using deep learning. A YOLOv8 model was trained and deployed to identify and extract license plates from images and videos. The application was developed using Python and the PyTorch framework, with a user-friendly interface provided by Streamlit.

## Key Features

- **Data Preparation:**
  - Utilized PyTorch and Python to process 433 pre-annotated images from the [Car Plate Detection dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) on Kaggle and converted annotations to YOLO format.

- **Model Training:**
  - The YOLOv8 model was trained in a Jupyter Notebook using the aforementioned dataset. The model achieved an mAP@0.5 of 0.8 and an mAP@0.5-0.95 of 0.55 over 100 epochs.

- **Deployment:**
  - Deployed the model for real-time license plate detection using Streamlit and Hugging Face Spaces at https://huggingface.co/spaces/Npps/Yolo_Car_Licence_Plate_Detection.

## Dataset

The dataset used for this project is the [Car Plate Detection dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) available on Kaggle. It contains images of vehicles with annotated license plates, which were used to train and validate the model.

## Code and Implementation

### Prerequisites

Ensure you have the following dependencies installed:

    pip install numpy pandas streamlit ultralytics opencv-python pytesseract

### Running the Application
To run the application, you need to have your trained YOLO model available and specify its path in the code. Here's a step-by-step guide:

1. Clone the Repository:
   ```bash
   git clone https://github.com/Npps1997/Automatic-License-Plate-Number-Detection.git
   cd Automatic-License-Plate-Number-Detection

2. Set up the Streamlit App:

Open the **'app.py'** file and replace **'path_to_your_trained_model'** with the actual path to your trained YOLO model.

3. Run the Application:

   ```bash
   streamlit run app.py

## Application Features
1. **Image Processing:**
  - Users can upload images, and the model will detect and highlight license plates, displaying confidence scores.

2. **Video Processing:**
  - Users can upload video files, and the application will process each frame to detect license plates, providing a video with detected plates highlighted.

3. **Real-Time Detection:**
  - The app supports real-time license plate detection from live video feeds.

## Model Training
The model training was conducted in a Jupyter Notebook, leveraging the flexibility and visualization capabilities of the notebook environment. The trained YOLOv8 model is utilized in this project to ensure efficient and accurate detection.

## Usage
### Uploading and Processing Media
* **Image Files:** Upload images in formats like JPG, PNG, BMP, etc.
* **Video Files:** Upload video files in formats like MP4, AVI, MOV, MKV, etc.
The app will display the detected license plates along with confidence scores. For images, the processed image will be shown; for videos, the processed video will be available for download.

## Additional Features
OCR (Optical Character Recognition): The project uses Tesseract OCR to read and display the text from detected license plates.

## License
This project is open-source and available under the MIT License.

## Acknowledgments
Thanks to the open-source community for the libraries and tools used in this project, including PyTorch, Streamlit, and Tesseract OCR.
