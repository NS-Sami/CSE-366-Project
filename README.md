#  Cotton Leaf Disease Detection
This project aims to automate the identification of various cotton leaf diseases using deep learning. By leveraging a pre-trained convolutional neural network (ResNet-50), the model classifies images of cotton leaves into one of seven disease categories. This type of classification is vital for timely and accurate disease diagnosis, which can significantly reduce crop losses in agriculture. The model is fine-tuned to adapt to cotton-specific disease features, improving its accuracy on real-world agricultural data.

Kaggle link - https://www.kaggle.com/code/nabi1subhan/pre-trained-cnn

## About Dataset
The dataset utilized in this project is SAR-CLD-2024: A Comprehensive Dataset for Cotton Leaf Disease Detection, which comprises a large collection of high-quality, augmented images of cotton leaves affected by various diseases. The dataset is systematically organized into seven distinct classes, with each class representing a specific type of leaf disease.

To ensure balanced representation, the data has been divided into training, validation, and test sets using stratified sampling, maintaining an even distribution across all classes. Furthermore, all images have been resized and normalized to meet the input specifications of the ResNet-50 architecture, ensuring optimal compatibility and performance during model training and evaluation.
Dataset link (kaggle) - https://www.kaggle.com/datasets/sabuktagin/dataset-for-cotton-leaf-disease-detection

## Tools Used
- Python

- PyTorch

- Torchvision

- Pandas & NumPy

- Matplotlib & Seaborn

- Scikit-learn

- Jupyter Notebook

- Kaggle GPU Kernel
