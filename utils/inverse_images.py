from PIL import Image, ImageOps
from pathlib import Path

# Créer le dossier de destination
output_dir = Path("mnist_digits_inverted")
output_dir.mkdir(exist_ok=True)

# Parcourir les dossiers 0-9
for digit in range(10):
    input_folder = Path(str(digit))
    output_folder = output_dir / str(digit)
    output_folder.mkdir(exist_ok=True)
    
    # Traiter chaque image dans le dossier
    for img_path in input_folder.glob("*.bmp"):
        img = Image.open(img_path)
        inverted_img = ImageOps.invert(img.convert("RGB"))
        inverted_img.save(output_folder / img_path.name)
        print(f"Inversé: {img_path} -> {output_folder / img_path.name}")

print("\nToutes les images ont été inversées avec succès!")
