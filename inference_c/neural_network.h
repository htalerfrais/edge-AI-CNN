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
void forward_pass(MLPModel *model, float *input, float *output);

/**
 * Fonction d'activation ReLU (Rectified Linear Unit)
 */
float relu(float x);

/**
 * Trouve l'indice de la valeur maximale (Prediction finale)
 */
int get_prediction(float *output);


#ifdef __cplusplus
}
#endif

#endif // NEURAL_NETWORK_H
