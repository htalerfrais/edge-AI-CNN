# ce fichier sert à exporter les poids des modèles pt générés avec les entrainement en pytorch
# pour les avoir dans un format exploitable par notre moteur d'inférence C

import torch
import torch.nn as nn
import numpy as np
from hector_train_mlp import MinimalMLP


def export_to_txt(model_path, output_path):
    # Chargement des poids PyTorch
    model = MinimalMLP()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    with open(output_path, 'w') as f:
        for name, param in model.named_parameters():
            data = param.detach().cpu().numpy().flatten()
            np.savetxt(f, data)
            
    print(f"Export terminé dans {output_path}")

if __name__ == "__main__":
    export_to_txt("models/mlp_model.pt", "models/mlp_model.txt")
    
