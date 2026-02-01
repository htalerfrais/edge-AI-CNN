#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "neural_network.h"


MLPModel* load_mlp_model(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Erreur lors de l'ouverture du fichier de poids");
        return NULL;
    }

    MLPModel *model = (MLPModel*)malloc(sizeof(MLPModel));
    model->W1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    model->b1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    model->W2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    model->b2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));

    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        if (fscanf(file, "%f", &model->W1[i]) != 1) break;
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        if (fscanf(file, "%f", &model->b1[i]) != 1) break;
    }
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        if (fscanf(file, "%f", &model->W2[i]) != 1) break;
    }
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


void forward_pass_mlp(MLPModel *model, float *input, float *output) {
    float hidden[HIDDEN_SIZE];

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float sum = model->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += model->W1[i * INPUT_SIZE + j] * input[j];
        }
        hidden[i] = relu(sum);
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float sum = model->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += model->W2[i * HIDDEN_SIZE + j] * hidden[j];
        }
        output[i] = sum;
    }
}


static int read_weights(FILE *file, float *buffer, int count) {
    for (int i = 0; i < count; i++) {
        if (fscanf(file, "%f", &buffer[i]) != 1) {
            return 0;
        }
    }
    return 1;
}

CNNModel* load_cnn_model(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Erreur lors de l'ouverture du fichier de poids");
        return NULL;
    }

    CNNModel *model = (CNNModel*)malloc(sizeof(CNNModel));
    if (!model) return NULL;

    model->conv1_w = (float*)malloc(C1_OUT_CH * C1_IN_CH * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    model->conv1_b = (float*)malloc(C1_OUT_CH * sizeof(float));
    model->conv2_w = (float*)malloc(C2_OUT_CH * C2_IN_CH * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    model->conv2_b = (float*)malloc(C2_OUT_CH * sizeof(float));
    model->fc_w    = (float*)malloc(OUTPUT_SIZE * FC_IN_FEATURES * sizeof(float));
    model->fc_b    = (float*)malloc(OUTPUT_SIZE * sizeof(float));

    int success = 1;
    success &= read_weights(file, model->conv1_w, C1_OUT_CH * C1_IN_CH * KERNEL_SIZE * KERNEL_SIZE);
    success &= read_weights(file, model->conv1_b, C1_OUT_CH);
    success &= read_weights(file, model->conv2_w, C2_OUT_CH * C2_IN_CH * KERNEL_SIZE * KERNEL_SIZE);
    success &= read_weights(file, model->conv2_b, C2_OUT_CH);
    success &= read_weights(file, model->fc_w, OUTPUT_SIZE * FC_IN_FEATURES);
    success &= read_weights(file, model->fc_b, OUTPUT_SIZE);

    fclose(file);

    if (!success) {
        fprintf(stderr, "Erreur fatale : Le fichier de poids %s est corrompu ou incomplet.\n", filename);
        free_cnn_model(model);
        return NULL;
    }

    printf("Modèle CNN chargé avec succès depuis %s\n", filename);
    return model;
}

void free_cnn_model(CNNModel *model) {
    if (model) {
        if (model->conv1_w) free(model->conv1_w);
        if (model->conv1_b) free(model->conv1_b);
        if (model->conv2_w) free(model->conv2_w);
        if (model->conv2_b) free(model->conv2_b);
        if (model->fc_w)    free(model->fc_w);
        if (model->fc_b)    free(model->fc_b);
        free(model);
    }
}

void forward_pass_cnn(CNNModel *model, float *input, float *output) {
    float layer1_out[C1_OUT_CH * C1_SIZE * C1_SIZE];
    
    for (int och = 0; och < C1_OUT_CH; och++) {
        for (int oh = 0; oh < C1_SIZE; oh++) {
            for (int ow = 0; ow < C1_SIZE; ow++) {
                float sum = model->conv1_b[och];
                int start_h = oh * 2 - 2; 
                int start_w = ow * 2 - 2;

                for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                    for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                        int in_h = start_h + kh;
                        int in_w = start_w + kw;
                        if (in_h >= 0 && in_h < 28 && in_w >= 0 && in_w < 28) {
                            float val = input[in_h * 28 + in_w];
                            float weight = model->conv1_w[och * (1 * 25) + kh * 5 + kw];
                            sum += val * weight;
                        }
                    }
                }
                layer1_out[och * (14 * 14) + oh * 14 + ow] = relu(sum);
            }
        }
    }

    float layer2_out[C2_OUT_CH * C2_SIZE * C2_SIZE];

    for (int och = 0; och < C2_OUT_CH; och++) {
        for (int oh = 0; oh < C2_SIZE; oh++) {
            for (int ow = 0; ow < C2_SIZE; ow++) {
                float sum = model->conv2_b[och];
                int start_h = oh * 2 - 2;
                int start_w = ow * 2 - 2;

                for (int ich = 0; ich < C1_OUT_CH; ich++) { 
                    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                        for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                            int in_h = start_h + kh;
                            int in_w = start_w + kw;
                            if (in_h >= 0 && in_h < 14 && in_w >= 0 && in_w < 14) {
                                float val = layer1_out[ich * (14 * 14) + in_h * 14 + in_w];
                                float weight = model->conv2_w[och * (16 * 25) + ich * 25 + kh * 5 + kw];
                                sum += val * weight;
                            }
                        }
                    }
                }
                layer2_out[och * (7 * 7) + oh * 7 + ow] = relu(sum);
            }
        }
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float sum = model->fc_b[i];
        for (int j = 0; j < FC_IN_FEATURES; j++) {
            sum += model->fc_w[i * FC_IN_FEATURES + j] * layer2_out[j];
        }
        output[i] = sum;
    }
}

float relu(float x) { 
    return x > 0 ? x : 0; 
}

int get_prediction(float *output) {
    int best_class = 0;
    float max_val = output[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            best_class = i;
        }
    }
    return best_class;
}


