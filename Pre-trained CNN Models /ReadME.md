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
