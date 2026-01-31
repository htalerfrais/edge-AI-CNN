import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from statistics import mean, median, stdev

# ============================
# Configuration
# ============================

IMAGE_DIR = "../data/"          # Folder containing MNIST-like images
MLP_MODEL_PATH = "./mlp.pt"
CNN_MODEL_PATH = "./cnn.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

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
    """
    model.eval()
    times = []
    correct = 0
    confidences = []

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
            if pred.item() == label:
                correct += 1

    return {
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

# ============================
# Main
# ============================

def main():
    print(f"Running benchmark on device: {DEVICE}")

    # Load models
    mlp = torch.load(MLP_MODEL_PATH, map_location=DEVICE)
    cnn = torch.load(CNN_MODEL_PATH, map_location=DEVICE)
    mlp.to(DEVICE)
    cnn.to(DEVICE)

    # Load dataset
    images = []
    labels = []

    for fname in sorted(os.listdir(IMAGE_DIR)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img = load_image(os.path.join(IMAGE_DIR, fname)).to(DEVICE)
            label = extract_label(fname)
            images.append(img)
            labels.append(label)

    print(f"Loaded {len(images)} images")

    # Run benchmarks
    mlp_results = benchmark_model(mlp, images, labels)
    cnn_results = benchmark_model(cnn, images, labels)

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


if __name__ == "__main__":
    main()