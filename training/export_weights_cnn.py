# de la même manière que cela est fait dans export_weight_mlp, ici, nous exportons 
# les poids du fichier .pt généré par l'entrainement du cnn
# dans un format .txt lisible par le loader du model en C de l'application.

import torch
import torch.nn as nn
import numpy as np
from train_cnn import CNN


def export_to_txt(model_path, output_path):
    # Chargement des poids PyTorch
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    with open(output_path, 'w') as f:
        for name, param in model.named_parameters():
            data = param.detach().cpu().numpy().flatten()
            np.savetxt(f, data)
    print(f"Export terminé dans {output_path}")

if __name__ == "__main__":
    export_to_txt("models/cnn_model.pt", "models/cnn_model.txt")

