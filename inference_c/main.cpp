// orchestre la capture de l'image raspberry avec OpenCV
// orchestre le prétraitement de l'image
// orchestre le chargement des poids dans la structure de neural network
// orchestre les prédictions faites par neural_network.c
// orchestre l'affichage sur l'écran


#include <iostream>
#include <vector>
#include "neural_network.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include "neural_network.h"

int main() {
    // 1. Chemins
    const char* model_path = "../training/Hector/models/mlp_model.txt";
    std::string image_path = "/Work/ProjectRepo/edge-AI-CNN/data/mnist_digit/3/digit_3_4.bmp";

    // 2. Chargement du modèle
    MLPModel* model = load_mlp_model(model_path);
    if (!model) return 1;

    // 3. Chargement et prétraitement de l'image avec OpenCV
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Erreur: Impossible de charger l'image." << std::endl;
        return 1;
    }

    // Redimensionnement en 28x28 si nécessaire
    cv::resize(img, img, cv::Size(28, 28));

    // Conversion en float et normalisation (-1.0 à 1.0 comme dans ton script Python)
    float input[INPUT_SIZE];
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            // MNIST original est souvent inversé (fond noir, chiffre blanc)
            // On normalise : (pixel / 255.0 - 0.5) / 0.5
            float pixel_val = (float)img.at<uchar>(i, j);
            input[i * 28 + j] = (pixel_val / 255.0f - 0.5f) / 0.5f;
        }
    }

    // 4. Inférence
    float output[OUTPUT_SIZE];
    forward_pass(model, input, output);

    // 5. Résultat
    int prediction = get_prediction(output);
    std::cout << "\nResultat pour l'image digit_3_4.bmp :" << std::endl;
    std::cout << ">> Chiffre predit : " << prediction << " <<" << std::endl;

    free_mlp_model(model);
    return 0;
}