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


    // 3. Prétraitement 
    cv::Mat resized;
    // Redimensionnement direct en 28x28
    cv::resize(img, resized, cv::Size(28, 28), 0, 0, cv::INTER_AREA);
    

    // 4. Normalisation [-1.0, 1.0] vers le tableau d'entrée
    float input[784];
    for (int i = 0; i < 784; i++) {
        float val = resized.data[i] / 255.0f;
        input[i] = (val - 0.1307f) / 0.3081f;
    }

    // 5. Inférence
    float scores[10];
    forward_pass(model, input, scores);
    
    for(int i = 0; i < 10; i++) {
        std::cout << "Score " << i << " = " << scores[i] << std::endl;
    }

    int result = get_prediction(scores);

    // 6. Affichage
    std::cout << "Fichier teste : " << imagePath << std::endl;
    std::cout << "Chiffre detecte : " << result << std::endl;

    // Nettoyage
    free_cnn_model(model);
    return 0;
}