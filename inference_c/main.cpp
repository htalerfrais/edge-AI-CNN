#include <opencv2/opencv.hpp>
#include <iostream>
#include "neural_network_cnn.h"

int main() {
    const char* imagePath = "../data/mnist_digit/2/digit_2_6.bmp";
    const char* modelPath = "cnn_model.txt";

    CNNModel* model = load_cnn_model(modelPath);
    if (!model) {
        std::cerr << "Erreur : Fichier de poids introuvable." << std::endl;
        return -1;
    }

    // 2. Chargement de l'image (en Niveaux de Gris)
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Erreur : Image introuvable." << std::endl;
        return -1;
    }

    // 3. Prétraitement Minimaliste (Pipeline MNIST)
    cv::Mat processed;
    // Inversion + Seuil automatique
    cv::threshold(img, processed, 128, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    // Redimensionnement vers 20x20
    cv::resize(processed, processed, cv::Size(20, 20));
    // Centrage dans un cadre 28x28 noir
    cv::Mat canvas = cv::Mat::zeros(28, 28, CV_8UC1);
    processed.copyTo(canvas(cv::Rect(4, 4, 20, 20)));

    // 4. Normalisation [-1.0, 1.0] vers le tableau d'entrée
    float input[784];
    for (int i = 0; i < 784; i++) {
        float val = canvas.data[i] / 255.0f;
        input[i] = (val - 0.5f) / 0.5f;
    }

    // 5. Inférence
    float scores[10];
    forward_pass(model, input, scores);
    int result = get_prediction(scores);

    // 6. Affichage
    std::cout << "Fichier teste : " << imagePath << std::endl;
    std::cout << "Chiffre detecte : " << result << std::endl;

    // Nettoyage
    free_cnn_model(model);
    return 0;
}