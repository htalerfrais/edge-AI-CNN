#include <stdio.h>
#include <stdlib.h>

#define D 784   // Input features
#define H 100   // Hidden layer neurons
#define C 10    // Output classes

// ReLU activation
float relu(float x) {
    return x > 0 ? x : 0;
}

int main() {
    FILE *fp = fopen("../models/mlp_weights.txt", "r");
    if (!fp) {
        printf("Error: cannot open weights file.\n");
        return 1;
    }

    // Allocate arrays
    float fc1_weight[H][D];
    float fc1_bias[H];
    float fc3_weight[C][H];
    float fc3_bias[C];

    // Read fc1 weights
    for(int i = 0; i < H; i++)
        for(int j = 0; j < D; j++)
            fscanf(fp, "%f", &fc1_weight[i][j]);

    // Read fc1 bias
    for(int i = 0; i < H; i++)
        fscanf(fp, "%f", &fc1_bias[i]);

    // Read fc3 weights
    for(int i = 0; i < C; i++)
        for(int j = 0; j < H; j++)
            fscanf(fp, "%f", &fc3_weight[i][j]);

    // Read fc3 bias
    for(int i = 0; i < C; i++)
        fscanf(fp, "%f", &fc3_bias[i]);

    fclose(fp);

    // Example input (flattened 28x28 image)
    float x[D];
    for(int i=0; i<D; i++)
        x[i] = 0.0; // replace with your actual input values

    // Forward pass: fc1 + ReLU
    float h[H];
    for(int i = 0; i < H; i++) {
        h[i] = fc1_bias[i];
        for(int j = 0; j < D; j++)
            h[i] += fc1_weight[i][j] * x[j];
        h[i] = relu(h[i]);
    }

    // Forward pass: fc3 (output layer)
    float y[C];
    for(int i = 0; i < C; i++) {
        y[i] = fc3_bias[i];
        for(int j = 0; j < H; j++)
            y[i] += fc3_weight[i][j] * h[j];
    }

    // Print output
    int pred = 0;
    printf("Output:\n");
    for(int i = 0; i < C; i++)
        {
        printf("%f ", y[i]);
        if(y[i] > pred) pred = i;
    printf("\n");
        }

    printf("Prédiction : %d\n", pred);

    return 0;
}