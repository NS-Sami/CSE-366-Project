Custom CNN – Cotton Leaf Disease Detection
A deep learning–based project for cotton leaf disease classification using a Custom Convolutional Neural Network (CNN).
This project focuses on building a CNN model from scratch, evaluating its performance, visualizing results, and analyzing key performance metrics such as accuracy, precision, recall, and F1-score.
Kaggle Notebook Link: Click Here
________________________________________
Features
•	Custom CNN model designed and implemented from scratch using PyTorch
•	Proper train, validation, and test split using stratified sampling
•	Data augmentation applied for better generalization
•	Visualization of model architecture using Torchinfo
•	Early stopping and model checkpointing for optimized training
•	Comprehensive performance evaluation using accuracy, precision, recall, and F1-score
•	Training and validation loss & accuracy curves for better insights
________________________________________
Dataset
The dataset contains 7 classes of cotton leaves:
•	Healthy Leaf
•	Curl Virus
•	Bacterial Blight
•	Leaf Redding
•	Leaf Hopper Jassids
•	Leaf Variegation
•	Herbicide Growth Damage
Dataset Link: Click Here
________________________________________
Tools & Libraries Used
•	Programming Language: Python
•	Deep Learning Framework: PyTorch, Torchvision
•	Data Analysis: Pandas, NumPy
•	Visualization: Matplotlib, Seaborn
•	Evaluation Metrics: Scikit-learn
•	Others: Torchinfo, Pillow, Jupyter Notebook, Kaggle GPU Kernel
________________________________________
Model Architecture Overview
•	3 Convolutional Layers with Batch Normalization & ReLU Activation
•	MaxPooling and AveragePooling for feature extraction
•	Fully Connected Layers with Dropout for regularization
•	Softmax activation for 7-class classification
________________________________________
Training Configuration
•	Batch Size: 64
•	Image Size: 128×128
•	Optimizer: Adam
•	Learning Rate: 0.001
•	Loss Function: Cross-Entropy with label smoothing
•	Epochs: 500
•	Early Stopping: Patience = 200
•	Hardware Used: Kaggle Tesla P100 GPU
________________________________________
Training Results
•	Best epoch achieved at 495
•	Training accuracy: 96.37%
•	Validation accuracy: 97.7%
•	Training loss: 0.6459
•	Validation loss: 0.533
________________________________________
Evaluation on Test Set
•	Final test accuracy: 97.8%
•	High precision, recall, and F1-score across all 7 classes
•	Balanced performance verified using confusion matrix
________________________________________
How to Run
•	Clone the repository
•	Install dependencies from requirements.txt
•	Train the model using the provided dataset
•	Evaluate the model on the test set
________________________________________
Future Enhancements
•	Integrate Grad-CAM visualization for explainable AI
•	Compare performance with ResNet50 and MobileNetV2
•	Deploy the model as a Flask / FastAPI web application
•	Extend dataset and test on real-world field images
