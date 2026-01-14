# ce fichier sert à exporter les poids des modèles pt générés avec les entrainement en pytorch
# pour les avoir dans un format exploitable par notre moteur d'inférence C

import torch
import torch.nn as nn
import numpy as np


# L'architecture doit être identique à votre MinimalMLP
class MinimalMLP(nn.Module):
    def __init__(self):
        super(MinimalMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

def export_to_txt(model_path, output_path):
    # Chargement des poids PyTorch
    model = MinimalMLP()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    with open(output_path, 'w') as f:
        for name, param in model.named_parameters():
            # On récupère les données sous forme de liste numpy
            data = param.detach().cpu().numpy()
            
            # Optionnel : écrire le nom et la forme pour le débug
            f.write(f"# {name} {data.shape}\n")
            
            # Sauvegarde des valeurs à plat (flatten) avec une précision fixe
            # 'fmt' peut être ajusté pour réduire la taille du fichier
            np.savetxt(f, data.flatten(), fmt='%f')
            
    print(f"Export terminé dans {output_path}")

if __name__ == "__main__":
    export_to_txt("models/mlp_model.pt", "models/mlp_model.txt")
    
