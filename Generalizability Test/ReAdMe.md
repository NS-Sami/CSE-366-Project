## Custom CNN – Generalizability Testing
A deep learning–based project to evaluate the generalization capability of a Custom Convolutional Neural Network (CNN) for cotton leaf disease detection.
The project focuses on training a CNN model from scratch, performing rigorous testing on a second unseen dataset, and visualizing model explainability using Grad-CAM.
________________________________________
## 📌 Features
•	Designed a Custom CNN architecture from scratch using PyTorch
•	Generalizability testing performed on a second dataset to evaluate robustness
•	Applied data augmentation techniques for better performance and reduced overfitting
•	Implemented stratified train-validation-test split (70:15:15)
•	Integrated Grad-CAM visualization for model interpretability
•	Analyzed performance using accuracy, precision, recall, F1-score, and confusion matrix
•	Optimized training using early stopping and checkpointing
________________________________________
## 📂 Dataset
Primary Dataset (Training & Validation)
•	Cotton leaf disease dataset with 4 classes:
o	Bacterial Blight
o	Healthy
o	Curl Virus
o	Fussarium Wilt
Secondary Dataset (Generalizability Testing)
•	Used an unseen dataset to test the robustness of the trained model
Dataset Link: Cotton Leaf Disease Dataset
________________________________________
## 🛠️ Tools & Libraries Used
•	Programming Language: Python
•	Deep Learning Frameworks: PyTorch, Torchvision
•	Explainability Tools: Grad-CAM, Torchinfo
•	Visualization Libraries: Matplotlib, Seaborn
•	Data Handling: Pandas, NumPy
•	Model Evaluation: Scikit-learn
•	Others: Pillow, OpenCV, tqdm, Jupyter Notebook, Kaggle GPU
________________________________________
## 🧠 Model Architecture Overview
•	3 Convolutional Layers with Batch Normalization and ReLU Activation
•	MaxPooling & AveragePooling layers for feature extraction
•	Fully Connected Layers with Dropout to reduce overfitting
•	Softmax activation for 4-class classification
________________________________________
## ⚙️ Training Configuration
•	Batch Size: 32 (Train), 8 (Validation/Test)
•	Image Size: 128×128
•	Optimizer: Adam
•	Learning Rate: 0.001
•	Loss Function: Cross-Entropy with label smoothing
•	Epochs: 500
•	Early Stopping: Patience = 200
•	Hardware Used: Kaggle Tesla P100 GPU
________________________________________
## 📊 Training Results
•	Best Training Accuracy: 96.4%
•	Best Validation Accuracy: 97.7%
•	Training Loss: 0.6459
•	Validation Loss: 0.533
•	Generalizability testing on unseen dataset achieved high accuracy and robust predictions
________________________________________
## ✅ Evaluation Metrics
•	Overall Test Accuracy: 97.8%
•	High precision, recall, and F1-score across all 4 classes
•	Confusion matrix visualization for better error analysis
________________________________________
## 🔍 Grad-CAM Visualization
•	Integrated Grad-CAM to visualize model attention regions
•	Helps understand why the model predicts a particular class
•	Enhances transparency and explainability of deep learning models
________________________________________

## 🔮 Future Enhancements
•	Compare performance with ResNet50, MobileNetV2, and VGG16
•	Deploy the model as a Flask / FastAPI web app
•	Test on real-world field images for better applicability
•	Enhance generalizability using larger and diverse datasets


