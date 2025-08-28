Cotton Leaf Disease Detection using a Hybrid CNN-SVM Model
This project implements a powerful hybrid classification model that combines a custom-built Convolutional Neural Network (CNN) with a Support Vector Machine (SVM) to accurately classify diseases in cotton leaves. The CNN acts as a sophisticated feature extractor, and the SVM performs the final classification.

Table of Contents
Project Overview

Model Architecture

Dataset

Technologies Used

Installation

Usage

Results

Contributing

License

Project Overview
The goal of this project is to automate the detection of various diseases affecting cotton plants by analyzing images of their leaves. Early and accurate diagnosis is crucial for effective crop management. We employ a two-stage machine learning pipeline where a deep learning model first learns to identify relevant visual features from the images, and a classical machine learning model then uses these features to make a final prediction.

Model Architecture
The hybrid model leverages the strengths of both deep learning and traditional machine learning:

Stage 1: Deep Feature Extraction (Custom CNN)
A custom Convolutional Neural Network is first trained on the cotton leaf image dataset. Instead of using this CNN for the final classification, we use its convolutional layers to learn a rich, hierarchical representation of the visual features (like textures, spots, colors, and edges) that distinguish different diseases.

Stage 2: Classification (SVM)
The trained CNN is then used as a feature extractor. Each image is passed through the CNN's convolutional base to generate a high-dimensional feature vector. This vector serves as the input for a Support Vector Machine (SVM), which is trained to find the optimal decision boundaries between the different disease classes in this feature space.

The workflow can be visualized as follows:

[Leaf Image] -> [Trained CNN Feature Extractor] -> [Feature Vector] -> [SVM Classifier] -> [Predicted Disease]

Dataset
This model is trained on the SAR-CLD-2024, a comprehensive dataset for cotton leaf disease detection.

Source: Dataset for Cotton Leaf Disease Detection on Kaggle

Total Images: 7,000

Data Split:

Training: 4,900 images (70%)

Validation: 1,050 images (15%)

Testing: 1,050 images (15%)

Classes (7):

Healthy Leaf

Bacterial Blight

Curl Virus

Herbicide Growth Damage

Leaf Hopper Jassids

Leaf Redding

Leaf Variegation

Technologies Used
Programming Language: Python

Deep Learning: PyTorch

Machine Learning: Scikit-learn

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Environment: Jupyter Notebook / Kaggle

Installation
To set up this project locally, clone the repository and install the necessary dependencies.

# 1. Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# 3. Install the required packages
pip install -r requirements.txt

Your requirements.txt file should include:

torch
torchvision
torchinfo
scikit-learn
pandas
numpy
matplotlib
seaborn
Pillow
tqdm

Usage
The project is structured as a Python notebook that follows these key steps:

Data Loading & Preprocessing: Load the dataset paths into a DataFrame and split it into training, validation, and test sets.

Data Augmentation: Apply transformations to the training data (e.g., random flips, rotations, color jitter) to improve model robustness.

CNN Training: Train the custom CNN model on the training set and use the validation set to find the best-performing model checkpoint.

Feature Extraction: Load the best CNN model and pass the training and test datasets through its feature layers to extract numerical feature vectors.

SVM Training & Evaluation: Train an SVM classifier using the extracted training features and evaluate its performance on the extracted test features.

Results
The hybrid approach demonstrated strong performance, with the SVM classifier effectively leveraging the features learned by the CNN.

CNN Performance: The standalone CNN achieved a validation accuracy of 87.1%.

Hybrid CNN-SVM Performance: The final hybrid model achieved a test accuracy of 90.5%.

Classification Report (Hybrid Model)
Class

Precision

Recall

F1-Score

Bacterial Blight

0.86

0.88

0.87

Curl Virus

0.90

0.79

0.84

Healthy Leaf

0.93

0.88

0.90

Herbicide Growth Damage

0.94

0.98

0.96

Leaf Hopper Jassids

0.85

0.93

0.89

Leaf Redding

0.87

0.88

0.88

Leaf Variegation

0.99

0.99

0.99

Accuracy





0.90

Macro Avg

0.91

0.90

0.90

Weighted Avg

0.91

0.90

0.90

Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

License
Distributed under the MIT License. See LICENSE file for more information.

