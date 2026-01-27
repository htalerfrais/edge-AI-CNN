// contient les prototypes de fonctions
// contient la structure necessaire pour stocker les poids du modèle

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

// Définition des dimensions du modèle MinimalMLP (utile pour l'allocation de mémoire)
#define INPUT_SIZE  784  // 28*28 pixels
#define HIDDEN_SIZE 512  // Couche intermédiaire
#define OUTPUT_SIZE 10   // Chiffres 0-9

// Structure pour regrouper les paramètres du réseau
typedef struct {
    float *W1; // Poids couche 1 (Taille: HIDDEN_SIZE * INPUT_SIZE)
    float *b1; // Biais couche 1 (Taille: HIDDEN_SIZE)
    float *W2; // Poids couche 2 (Taille: OUTPUT_SIZE * HIDDEN_SIZE)
    float *b2; // Biais couche 2 (Taille: OUTPUT_SIZE)
} MLPModel;

/**
 * Alloue la mémoire et charge les poids depuis le fichier .txt
 * @param filename Chemin vers mlp_model.txt
 * @return Un pointeur vers la structure MLPModel remplie
 */
MLPModel* load_mlp_model(const char *filename);

/**
 * Libère la mémoire allouée pour le modèle
 */
void free_mlp_model(MLPModel *model);

/**
 * Effectue la passe directe (Forward Pass)
 * @param model Le modèle chargé
 * @param input L'image prétraitée (784 floats)
 * @param output Tableau pour stocker les probabilités de sortie (10 floats)
 */
void forward_pass_mlp(MLPModel *model, float *input, float *output);

/**
 * Fonction d'activation ReLU (Rectified Linear Unit)
 */
float relu(float x);

/**
 * Trouve l'indice de la valeur maximale (Prediction finale)
 */
int get_prediction(float *output);

// --- Constantes de l'architecture ---
#define IMG_SIZE      28
#define KERNEL_SIZE   5

// Layer 1 : Conv (Input 1ch 28x28 -> Output 16ch 14x14 car stride 2)
#define C1_IN_CH      1
#define C1_OUT_CH     16
#define C1_SIZE       14

// Layer 2 : Conv (Input 16ch 14x14 -> Output 32ch 7x7 car stride 2)
#define C2_IN_CH      16
#define C2_OUT_CH     32
#define C2_SIZE       7

// Layer 3 : Fully Connected (Flatten 32*7*7 = 1568 -> 10)
#define FC_IN_FEATURES (C2_OUT_CH * C2_SIZE * C2_SIZE) // 1568
#define OUTPUT_SIZE    10


// --- Structure du modèle ---
typedef struct {
    // Poids et Biais de la Couche Conv 1
    // Taille : 16 filtres * 1 canal * 5 * 5
    float *conv1_w; 
    float *conv1_b;

    // Poids et Biais de la Couche Conv 2
    // Taille : 32 filtres * 16 canaux * 5 * 5
    float *conv2_w;
    float *conv2_b;

    // Poids et Biais de la Couche Dense (FC)
    // Taille : 10 classes * 1568 features
    float *fc_w;
    float *fc_b;
} CNNModel;


// Alloue et charge les poids depuis le fichier .txt
CNNModel* load_cnn_model(const char *filename);


//Libère la mémoire du modèle
void free_cnn_model(CNNModel *model);


//Effectue l'inférence complète
void forward_pass_cnn(CNNModel *model, float *input, float *output);

// Softmax (Optionnel pour l'inférence, mais utile pour des scores propres)
void softmax(float *input, int size);


#ifdef __cplusplus
}
#endif

#endif // NEURAL_NETWORK_H
