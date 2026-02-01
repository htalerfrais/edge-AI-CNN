#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

#define INPUT_SIZE  784
#define HIDDEN_SIZE 512
#define OUTPUT_SIZE 10

typedef struct {
    float *W1;
    float *b1;
    float *W2;
    float *b2;
} MLPModel;

MLPModel* load_mlp_model(const char *filename);
void free_mlp_model(MLPModel *model);
void forward_pass_mlp(MLPModel *model, float *input, float *output);
float relu(float x);
int get_prediction(float *output);

#define IMG_SIZE      28
#define KERNEL_SIZE   5
#define C1_IN_CH      1
#define C1_OUT_CH     16
#define C1_SIZE       14
#define C2_IN_CH      16
#define C2_OUT_CH     32
#define C2_SIZE       7
#define FC_IN_FEATURES (C2_OUT_CH * C2_SIZE * C2_SIZE)
#define OUTPUT_SIZE    10

typedef struct {
    float *conv1_w; 
    float *conv1_b;
    float *conv2_w;
    float *conv2_b;
    float *fc_w;
    float *fc_b;
} CNNModel;

CNNModel* load_cnn_model(const char *filename);
void free_cnn_model(CNNModel *model);
void forward_pass_cnn(CNNModel *model, float *input, float *output);
void softmax(float *input, int size);


#ifdef __cplusplus
}
#endif

#endif // NEURAL_NETWORK_H
