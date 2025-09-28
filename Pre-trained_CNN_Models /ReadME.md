## 🌿 Cotton Leaf Disease Detection
This repository contains two deep learning–based models — ResNet50 and VGG16 — for detecting 7 types of cotton leaf diseases using the SAR-CLD-2024 dataset.

The models are trained, fine-tuned, and evaluated to achieve high accuracy for practical agricultural applications.
________________________________________
## 📌 Project Overview

•	Develops AI-powered disease detection for cotton leaves using deep learning.

•	Implements two approaches:

o	ResNet50 – Transfer learning with a pre-trained CNN

o	VGG16 – Custom CNN fine-tuned for cotton leaf classification

•	Automates early detection to support farmers and improve crop yield.
________________________________________
## 📂 Dataset Details

•	Dataset Name: SAR-CLD-2024

•	Total Images: 7,000

•	Number of Classes: 7

o	Healthy Leaf

o	Leaf Hopper Jassids

o	Leaf Redding

o	Curl Virus

o	Herbicide Growth Damage

o	Bacterial Blight

o	Leaf Variegation

## •	Data Split:

o	Training → 70%

o	Validation → 15%

o	Testing → 15%

## •	Source: Kaggle – SAR-CLD-2024 Dataset
________________________________________
## 🧠 Model Details
## 🔹 Model 1: ResNet50 (Transfer Learning)

•Architecture: Pre-trained ResNet50

•	Input Size: 224 × 224

•	Optimizer: Adam

•	Loss Function: CrossEntropy

•	Batch Size: 32

•	Epochs: 10

•	Accuracy:

o	Training → ~99%

o	Validation → ~96%

o	Testing → ~95%
________________________________________
## 🔹 Model 2: VGG16 (Custom CNN)

•	Architecture: Modified VGG16-based CNN

•	Input Size: 128 × 128

•	Optimizer: Adam (learning rate = 0.0001)

•	Loss Function: CrossEntropy

•	Batch Size: 32 (train), 8 (val/test)

•	Epochs: 20

•	Device Used: Tesla P100 GPU

•	Accuracy:

o	Training → ~77.3%

o	Validation → ~74.0%

o	Testing → ~76.0%

•	Per-Class Accuracy Highlights:

o	Curl Virus → 91.3%

o	Leaf Variegation → 82.7%

o	Herbicide Damage → 77.3%

o	Healthy Leaf → 74.7%

o	Leaf Redding → 71.3%

o	Bacterial Blight → 73.3%

o	Leaf Hopper Jassids → 61.3%
________________________________________
## 🚀 Usage
## ResNet50 Model

•	Train the model: python train.py

•	Test the model: python test.py

•	Predict single image: python predict.py --image path_to_image.jpg

## VGG16 Model

•	Train the model: python train.py

•	Evaluate on test set: python test.py

•	Predict single image: python predict.py --image path_to_image.jpg
________________________________________

## 📊 Results Summary

•	## ResNet50 Model:

o	High overall accuracy (~95%)

o	Better generalization due to transfer learning

## •	 VGG16 Model:

o	Achieved ~76% test accuracy

o	Strong performance on Curl Virus and Leaf Variegation classes

## •	 Comparison Insight:

o	ResNet50 performs better overall

o	VGG16 still provides competitive results for smaller input sizes

