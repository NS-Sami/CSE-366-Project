## Cotton Leaf Disease Detection using a Hybrid CNN-SVM Model
A hybrid deep learning and machine learning–based project for cotton leaf disease classification. This model combines a custom Convolutional Neural Network (CNN) for feature extraction and a Support Vector Machine (SVM) for classification, achieving high accuracy and robustness.
________________________________________
## Features

•	Custom CNN architecture designed for deep feature extraction

•	SVM classifier for final disease prediction

•	Hybrid approach combining deep learning and machine learning

•	Data augmentation for improving model generalization

•	Trained and evaluated on a comprehensive cotton leaf disease dataset

•	Detailed performance analysis with accuracy, precision, recall, and F1-score
________________________________________
## Project Overview
The goal of this project is to automate the detection of various diseases affecting cotton plants by analyzing images of their leaves.
Early and accurate diagnosis is essential for effective crop management and improving yield.
The pipeline follows a two-stage approach:

•	Stage 1 – Feature Extraction (Custom CNN):
The CNN learns complex visual features like textures, spots, colors, and edges.

•	Stage 2 – Classification (SVM):
The CNN-extracted features are fed into an SVM classifier, which finds the optimal decision boundaries for disease classification.

## Workflow:
[Leaf Image] → [CNN Feature Extractor] → [Feature Vector] → [SVM Classifier] → [Predicted Disease]
________________________________________
## Dataset

•	Dataset Name: SAR-CLD-2024

•	Source: Kaggle – Dataset for Cotton Leaf Disease Detection

•	Total Images: 7,000

•	Data Split:

o	Training: 4,900 images (70%)

o	Validation: 1,050 images (15%)

o	Testing: 1,050 images (15%)

•	Classes (7):

o	Healthy Leaf

o	Bacterial Blight

o	Curl Virus

o	Herbicide Growth Damage

o	Leaf Hopper Jassids

o	Leaf Redding

o	Leaf Variegation
________________________________________
## Technologies Used

•	Programming Language: Python

•	Deep Learning Framework: PyTorch

•	Machine Learning Library: Scikit-learn

•	Data Handling: Pandas, NumPy

•	Visualization: Matplotlib, Seaborn

•	Development Environment: Jupyter Notebook / Kaggle
________________________________________
## Usage

This project follows the steps below:

1.	Data Loading & Preprocessing

o	Load dataset paths and split them into training, validation, and test sets.

2.	Data Augmentation

o	Apply transformations like random flips, rotations, and color adjustments to improve robustness.

3.	CNN Training

o	Train the custom CNN model and select the best-performing checkpoint using the validation set.

4.	Feature Extraction

o	Use the trained CNN to extract feature vectors from leaf images.

5.	SVM Training & Evaluation

o	Train the SVM classifier on extracted CNN features.

o	Evaluate model performance on the test set.
________________________________________
## Results
The hybrid CNN-SVM model significantly outperforms a standalone CNN:

•	Standalone CNN Accuracy: 87.1%

•	Hybrid CNN-SVM Accuracy: 90.5%
Performance Highlights:

•	High precision and recall across all disease classes

•	Superior generalization due to CNN feature extraction + SVM classification
________________________________________


