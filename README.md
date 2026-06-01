# EcoGrow Platform

EcoGrow is a precision agriculture web platform built using Flask and machine learning to assist farmers in making informed decisions. It helps determine the best crop to plant, suggests fertilizers, identifies crop diseases using image recognition, and provides cures. The platform is designed with an intuitive frontend and animated visuals to enhance user engagement and accessibility.

## Features

- Crop Recommendation based on soil type, temperature, humidity, and rainfall
- Fertilizer Suggestion to improve crop yield based on soil and crop type
- Plant Disease Detection using deep learning image classification
- Disease Curing Guidelines to support farmers in treating affected crops
- Responsive Web UI with custom animations and a visually engaging experience

## Tech Stack

| Layer        | Technology                         |
|--------------|------------------------------------|
| Frontend     | HTML5, CSS3, Bootstrap, JavaScript |
| Backend      | Python Flask                       |
| ML Models    | Scikit-learn (Crop & Fertilizer), CNN (Disease) |
| Image Models | Trained using TensorFlow/Keras     |
| Deployment   | Localhost                          |

## Project Structure

```
EcoGrow-Platform/
├── templates/              # HTML Templates
│   ├── base.html
│   ├── crop.html
│   ├── disease.html
│   └── ...
├── static/                 # Static files (CSS, JS, images)
│   └── images/
├── models/                 # Pre-trained ML models (pkl/h5 files)
├── app.py                  # Flask app entry point
├── crop_recommendation.py  # Crop prediction logic
├── fertilizer.py           # Fertilizer prediction logic
├── disease_prediction.py   # Plant disease prediction logic
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Getting Started

### Prerequisites

Make sure you have Python 3.7 or later installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

### Run the Application

To start the Flask app locally:

```bash
python app.py
```

Then open your browser and navigate to: http://localhost:5000

## Screenshots

Below are some screenshots of the platform:

![Home](https://github.com/user-attachments/assets/2378500a-fe75-4344-9135-ff458f3908b4)
![crop](https://github.com/user-attachments/assets/041f5889-9ae5-4070-bf14-91cccc265942)
![crop-result](https://github.com/user-attachments/assets/12728206-404b-41c8-906c-886a674f4444)
![fertilizer](https://github.com/user-attachments/assets/29688dc0-225c-43ce-95e8-7347b6d4441d)
![fertilizer-result](https://github.com/user-attachments/assets/84b0f8d5-c425-46bc-8fc4-43bd2cd8d581)
![disease](https://github.com/user-attachments/assets/2d1b4a28-e22b-49e0-8c92-09ff791f3e5e)
![disease-result](https://github.com/user-attachments/assets/122b3fc7-1e39-40eb-b992-6d731a8a2fec)

- Home Page  
- About Us and Services  
- Crop Recommendation Form  
- Crop Result Page  
- Fertilizer Input Page  
- Fertilizer Suggestion Page  
- Disease Detection Upload  
- Disease Diagnosis and Cure Information

## Machine Learning Models

### 1. Crop Recommendation
- Inputs: N, P, K values, temperature, humidity, pH, rainfall
- Model: Random Forest Classifier
- Dataset: Collected from Kaggle and agricultural databases

### 2. Fertilizer Suggestion
- Logic: Custom condition-based mapping using soil nutrients and crop type

### 3. Disease Detection
- Model: Convolutional Neural Network trained on PlantVillage dataset
- Classes: Tomato leaf diseases (e.g., Early Blight, Late Blight, Leaf Mold)
- Input: Leaf image uploaded by the user

## Routes and Pages

| Route         | Description                            |
|---------------|----------------------------------------|
| `/`           | Home Page                              |
| `/crop`       | Crop Recommendation Page               |
| `/fertilizer` | Fertilizer Suggestion Page             |
| `/disease`    | Disease Detection Page (Image Upload)  |
| `/about`      | About Us Page with animations          |




## Contact

- GitHub: [@Nish-011-100](https://github.com/Nish-011-100)

## Acknowledgments

- PlantVillage dataset for disease classification
- Kaggle datasets for crop and fertilizer recommendations
- Bootstrap and open-source animation assets for UI
