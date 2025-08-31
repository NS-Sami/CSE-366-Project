#  Cotton Leaf Disease Detection
A deep learning–based project for cotton leaf disease classification using Custom CNN, Hybrid CNN + ML Classifiers, and Pretrained CNN models (ResNet, MobileNetV2).
This project also includes Grad-CAM visualization and Generalizability Testing to evaluate robustness on a second dataset.

Kaggle links -
  CustmCNN - https://www.kaggle.com/code/sheikhrafi/customtest
  Generalizability Test - https://www.kaggle.com/code/nabi1subhan/generalizability-test
  Hybrid Models -
    SVM - https://www.kaggle.com/code/nabi1subhan/test-2-0-hybrid-svm
    KNN - https://www.kaggle.com/code/sam1gaming/test-2-0-hybrid-knn
  Pre-train CNN Models -
    Resnet50 - https://www.kaggle.com/code/nabi1subhan/pre-trained-cnn-resnet
    VGG16 - 

## Features
Custom CNN model designed from scratch

Hybrid model (CNN feature extractor + ML classifiers)

Pretrained models (ResNet, MobileNetV2) with transfer learning

Grad-CAM visualization for explainability

Generalizability testing on a second dataset

Performance evaluation with Accuracy, Class Accuracy, Precision, Recall, F1-score

To ensure balanced representation, the dataset was divided into training, validation, and test sets using stratified sampling, preserving an even distribution across all classes. All images were resized and normalized according to the input requirements of the selected model architectures (Custom CNN, Hybrid CNN+ML classifiers, ResNet50, and VGG16), ensuring consistency and optimal performance during training and evaluation.

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
