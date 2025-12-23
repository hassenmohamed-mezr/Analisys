import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# -------------------------
# تحميل الصور
# -------------------------
def load_images(root="Data", limit=None):
    X, y = [], []

    for lbl in ["0", "1"]:
        folder = os.path.join(root, lbl)
        files = os.listdir(folder)

        if limit is not None:
            files = files[:limit]

        for f in files:
            path = os.path.join(folder, f)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (32, 32))
            img = img.flatten() / 255.0   

            X.append(img)
            y.append(int(lbl))

    return np.array(X), np.array(y)

def preprocess_data(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X


# ---------- 3. Train & Evaluate ----------
def train_and_evaluate(X, y, layers, activation):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MLPClassifier(
        hidden_layer_sizes=layers,
        activation=activation,
        max_iter=500,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return acc, cm


# ---------- 4. Plot Confusion Matrix ----------
def plot_confusion_matrix(cm, title):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


    # ---------- 5. Run Experiment ----------
def run_experiment(layers, activation):
    X, y = load_images("Data" , limit=100)
    X = preprocess_data(X)

    acc, cm = train_and_evaluate(X, y, layers, activation)

    print(f"Layers: {layers}")
    print(f"Activation: {activation}")
    print(f"Accuracy: {acc:.4f}")

    plot_confusion_matrix(cm, f"{activation} | Layers {layers}")




# ---------- 6. Experiments ----------
run_experiment((64,), 'relu')
run_experiment((128, 64), 'relu')

run_experiment((64,), 'tanh')
run_experiment((128, 64), 'tanh')

run_experiment((64,), 'logistic')
run_experiment((128, 64), 'logistic')    