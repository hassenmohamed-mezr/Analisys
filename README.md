🎗️ Breast Cancer Image Classifier

A simple machine learning project for classifying breast tissue images as benign or malignant.

---

## 🧠 Idea
The model learns patterns from grayscale microscope images and predicts:
- 0 → Benign
- 1 → Malignant

---

## ⚙️ Approach
- Resize images to 32×32
- Convert to grayscale
- Normalize pixel values
- Flatten images into feature vectors
- Train an MLP (Neural Network)

---

## 🧪 Experiments
Multiple network designs were tested to study:
- Effect of **number of layers**
- Effect of **activation functions**

Activations used:
- ReLU
- Tanh
- Sigmoid

---

## 📊 Evaluation
- Accuracy score
- Confusion matrix  
  (special focus on **missed cancer cases**, not just accuracy)

---

## ▶️ Run
```bash
pip install numpy opencv-python scikit-learn matplotlib seaborn
python breast_cancer_classifier.py
