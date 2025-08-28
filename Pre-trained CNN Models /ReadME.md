🌿 Cotton Leaf Disease Detection (ResNet50)
📌 Overview

A Deep Learning-based Image Classification project that uses a Pre-trained ResNet50 CNN to detect 7 types of cotton leaf diseases. The model is fine-tuned using the SAR-CLD-2024 dataset and achieves high accuracy.

📂 Dataset

Dataset: SAR-CLD-2024
Total Images: 7000
Classes (7): Healthy, Leaf Hopper Jassids, Leaf Redding, Curl Virus, Herbicide Damage, Bacterial Blight, Leaf Variegation
Split: Train 70% • Val 15% • Test 15%

🧠 Model Details

Architecture: ResNet50 (Pre-trained on ImageNet)

Input Size: 224×224

Optimizer: Adam

Loss: CrossEntropy

Batch Size: 32

Epochs: 10

⚙️ Installation
git clone https://github.com/your-username/Cotton-Leaf-Disease-Detection.git
cd Cotton-Leaf-Disease-Detection
pip install -r requirements.txt

🚀 Usage

Train the model:

python train.py


Test the model:

python test.py


Predict single image:

python predict.py --image path_to_image.jpg
📊 Results
Dataset	Accuracy
Training	~99%
Validation	~96%
Testing	~95%

Project Structure
Cotton-Leaf-Disease-Detection/
│── data/           # Dataset
│── checkpoints/    # Saved models
│── train.py        # Training script
│── test.py         # Testing script
│── predict.py      # Prediction script
│── requirements.txt
│── README.md



🌿 Cotton Leaf Disease Detection using VGG16
📌 Overview

This project implements a Deep Learning-based image classification system for detecting 7 cotton leaf conditions using a VGG16-based CNN model.
The model is trained on the SAR-CLD-2024 dataset and achieves 76% test accuracy.

📂 Dataset

Dataset: SAR-CLD-2024
Total Images: 7000
Classes (7):

Healthy Leaf

Leaf Hopper Jassids

Leaf Redding

Curl Virus

Herbicide Growth Damage

Bacterial Blight

Leaf Variegation

Split: Train 70% | Validation 15% | Test 15%

🧠 Model Details

Architecture: VGG16-based Custom CNN

Input Size: 128×128

Optimizer: Adam (lr=0.0001)

Loss: CrossEntropy

Batch Size: 32 (train), 8 (val/test)

Epochs: 20

Device Used: Tesla P100 GPU

⚙️ Installation
git clone https://github.com/your-username/Cotton-Leaf-Disease-Detection-VGG16.git
cd Cotton-Leaf-Disease-Detection-VGG16
pip install -r requirements.txt

🚀 Usage

Train the model:

python train.py


Evaluate on test set:

python test.py


Predict single image:

python predict.py --image path_to_image.jpg

📊 Results
Metric	Training	Validation	Testing
Accuracy	77.3%	74.0%	76.0%
Loss	0.59	0.70	0.71

Per-Class Accuracy:

Class	Accuracy
Healthy Leaf	74.7%
Leaf Hopper Jassids	61.3%
Leaf Redding	71.3%
Curl Virus	91.3%
Herbicide Damage	77.3%
Bacterial Blight	73.3%
Leaf Variegation	82.7%
📁 Project Structure
Cotton-Leaf-Disease-Detection-VGG16/
│── data/             # Dataset
│── checkpoints/      # Trained model weights
│── train.py          # Training script
│── test.py           # Testing script
│── predict.py        # Single image prediction
│── utils.py          # Helper functions
│── requirements.txt  # Dependencies
│── README.md         # Documentation
