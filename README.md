# HOG + SVM Image Classifier

This project implements an image classification system using a custom implementation of the Histogram of Oriented Gradients (HOG) feature extractor and a Support Vector Machine (SVM) classifier.

The system is trained to classify images into three categories:
- 🐱 Cats  
- 🐶 Dogs  
- 🧑 Humans  

---

## 🧠 Features

- Manual HOG implementation (no external feature libraries)
- Gradient magnitude and orientation calculation
- Cell histograms with bilinear interpolation
- Block normalization
- SVM training using scikit-learn
- Classifies unseen test images
- Outputs HOG feature vectors to a text file
