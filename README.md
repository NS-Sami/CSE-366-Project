#  Cotton Leaf Disease Detection
This project aims to automate the identification of various cotton leaf diseases using deep learning. By leveraging a pre-trained convolutional neural network (ResNet-50), the model classifies images of cotton leaves into one of seven disease categories. This type of classification is vital for timely and accurate disease diagnosis, which can significantly reduce crop losses in agriculture. The model is fine-tuned to adapt to cotton-specific disease features, improving its accuracy on real-world agricultural data.

Kaggle link - https://www.kaggle.com/code/nabi1subhan/pre-trained-cnn

## About Dataset
The dataset used in this project is the SAR-CLD-2024: A Comprehensive Dataset for Cotton Leaf Disease Detection, which includes thousands of high-quality, augmented images of diseased cotton leaves. The dataset is organized into seven distinct classes, each representing a different type of leaf disease. The data is split into training, validation, and test sets using stratified sampling to ensure all classes are evenly distributed. All images were resized and normalized to fit the ResNet-50 architecture's input requirements.

Dataset link (kaggle) - https://www.kaggle.com/datasets/sabuktagin/dataset-for-cotton-leaf-disease-detection

## Tools Used
Python – Core programming language used for data processing and modeling
PyTorch – Deep learning framework used for building and training the CNN
Torchvision – For importing and modifying pretrained ResNet-50
Pandas & NumPy – Data manipulation and preprocessing
Matplotlib & Seaborn – Data visualization and plotting
Scikit-learn – Evaluation metrics and classification reports
Jupyter Notebook – Interactive environment to write and test the code
Kaggle GPU Kernel – For training on GPU-accelerated hardware
