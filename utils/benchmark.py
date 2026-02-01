import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from statistics import mean, median, stdev

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ============================
# Configuration
# ============================

IMAGE_DIR = "../data/mnist_digit_test"          # Folder containing MNIST-like images
MLP_MODEL_PATH = "../training/models/mlp_model.pt"
CNN_MODEL_PATH = "../training/models/cnn_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# ============================
# Models
# ============================

class MinimalMLP(nn.Module):
    def __init__(self):
        super(MinimalMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)
    

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.num_classes = 10
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU())
        self.fc = nn.Linear(7*7*32, self.num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out.flatten(start_dim = 1))
        return out


# ============================
# Utilities
# ============================

def load_image(path):
    """
    Load a 28x28 MNIST-like image and apply MNIST normalization.
    """
    img = Image.open(path).convert("L")
    img = img.resize((28, 28))
    img = np.array(img, dtype=np.float32) / 255.0
    img = (img - MNIST_MEAN) / MNIST_STD
    return torch.tensor(img).unsqueeze(0).unsqueeze(0)  # [1,1,28,28]


def extract_label(filename):
    """
    Assumes filenames like: 5_001.png or 5.png
    """
    return int(filename.split("_")[0])


def benchmark_model(model, images, labels):
    """
    Runs inference on all images and collects benchmark metrics.
    Also returns predicted labels for confusion matrix plotting.
    """
    model.eval()
    times = []
    correct = 0
    confidences = []
    preds_list = []

    with torch.no_grad():
        for img, label in zip(images, labels):
            start = time.perf_counter()
            output = model(img)
            end = time.perf_counter()

            elapsed_ms = (end - start) * 1000
            times.append(elapsed_ms)

            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)

            confidences.append(conf.item())
            preds_list.append(pred.item())
            if pred.item() == label:
                correct += 1

    metrics = {
        "accuracy": correct / len(images),
        "mean_time_ms": mean(times),
        "median_time_ms": median(times),
        "std_time_ms": stdev(times) if len(times) > 1 else 0.0,
        "min_time_ms": min(times),
        "max_time_ms": max(times),
        "throughput_fps": 1000.0 / mean(times),
        "mean_confidence": mean(confidences),
        "samples": len(images),
    }

    return metrics, preds_list


def plot_confusion_matrix(labels, preds, title="Confusion Matrix"):
    cm = confusion_matrix(labels, preds, labels=list(range(10)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

# ============================
# Main
# ============================

def main():
    print(f"Running benchmark on device: {DEVICE}")

    # Load models
    mlp = MinimalMLP()
    cnn = CNN()
    mlp.load_state_dict(torch.load(MLP_MODEL_PATH))
    cnn.load_state_dict(torch.load(CNN_MODEL_PATH))
    mlp.eval()
    cnn.eval()
    mlp.to(DEVICE)
    cnn.to(DEVICE)

    # Load dataset
    images = []
    labels = []

    for class_name in sorted(os.listdir(IMAGE_DIR)):
        class_path = os.path.join(IMAGE_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        for fname in sorted(os.listdir(class_path)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", "bmp")):
                img_path = os.path.join(class_path, fname)
                img = load_image(img_path).to(DEVICE)
                images.append(img)
                labels.append(int(class_name))  # use folder name as label

    print(f"Loaded {len(images)} images")

    # Run benchmarks
    mlp_results, mlp_preds = benchmark_model(mlp, images, labels)
    cnn_results, cnn_preds = benchmark_model(cnn, images, labels)

    # Display results
    print("\n===== BENCHMARK RESULTS =====\n")

    def print_results(name, r):
        print(f"{name}")
        print("-" * len(name))
        print(f"Accuracy              : {r['accuracy']*100:.2f}%")
        print(f"Mean inference time   : {r['mean_time_ms']:.3f} ms")
        print(f"Median inference time : {r['median_time_ms']:.3f} ms")
        print(f"Std inference time    : {r['std_time_ms']:.3f} ms")
        print(f"Min / Max time        : {r['min_time_ms']:.3f} / {r['max_time_ms']:.3f} ms")
        print(f"Throughput            : {r['throughput_fps']:.1f} FPS")
        print(f"Mean confidence       : {r['mean_confidence']:.3f}")
        print(f"Samples               : {r['samples']}")
        print()

    print_results("MLP", mlp_results)
    print_results("CNN", cnn_results)


    # Plot confusion matrices
    plot_confusion_matrix(labels, mlp_preds, title="MLP Confusion Matrix")
    plot_confusion_matrix(labels, cnn_preds, title="CNN Confusion Matrix")

if __name__ == "__main__":
    main()