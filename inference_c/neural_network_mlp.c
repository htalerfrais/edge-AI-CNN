#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "neural_network_mlp.h"


MLPModel* load_mlp_model(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Erreur lors de l'ouverture du fichier de poids");
        return NULL;
    }

    // Allocation de la structure principale
    MLPModel *model = (MLPModel*)malloc(sizeof(MLPModel));

    // Allocation de la mémoire pour chaque couche via les dimensions renseignées dans neural_network.h
    model->W1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    model->b1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    model->W2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    model->b2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));

    // Lecture des Poids Couche 1
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        if (fscanf(file, "%f", &model->W1[i]) != 1) break;
    }

    // Lecture des Biais Couche 1
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        if (fscanf(file, "%f", &model->b1[i]) != 1) break;
    }

    // Lecture des Poids Couche 2
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        if (fscanf(file, "%f", &model->W2[i]) != 1) break;
    }

    // Lecture des Biais Couche 2
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (fscanf(file, "%f", &model->b2[i]) != 1) break;
    }

    fclose(file);
    printf("Modèle chargé avec succès depuis %s\n", filename);
    return model;
}

void free_mlp_model(MLPModel *model) {
    if (model) {
        free(model->W1);
        free(model->b1);
        free(model->W2);
        free(model->b2);
        free(model);
    }
}



// Fonction d'activation ReLU : f(x) = max(0, x)
float relu(float x) {
    return x > 0 ? x : 0;
}


void forward_pass(MLPModel *model, float *input, float *output) {
    // 1. Couche Cachée (Hidden Layer) : 512 neurones
    float hidden[HIDDEN_SIZE];

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        // Initialisation avec le biais b1
        float sum = model->b1[i];
        
        // Produit scalaire : Poids W1 * Entrée
        for (int j = 0; j < INPUT_SIZE; j++) {
            // W1 est stocké à plat : index = i * largeur + j
            sum += model->W1[i * INPUT_SIZE + j] * input[j];
        }
        
        // Application de l'activation ReLU
        hidden[i] = relu(sum);
    }

    // 2. Couche de Sortie (Output Layer) : 10 neurones
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        // Initialisation avec le biais b2
        float sum = model->b2[i];
        
        // Produit scalaire : Poids W2 * Sortie de la couche cachée
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += model->W2[i * HIDDEN_SIZE + j] * hidden[j];
        }
        
        // Stockage du résultat final (Scores/Logits)
        output[i] = sum;
    }
}

int get_prediction(float *output) {
    int best_class = 0;
    float max_val = output[0];
    
    // recherche de max parmis toutes les output classes
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            best_class = i;
        }
    }
    return best_class;
}