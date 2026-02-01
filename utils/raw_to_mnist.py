import cv2
import numpy as np
import os

def convert_to_mnist(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Image not found or path is incorrect")
    
    if np.mean(img) > 127:
        img = 255 - img
    
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y:y+h, x:x+w]
    
    if w > h:
        new_w = 20
        new_h = int(h * (20 / w))
    else:
        new_h = 20
        new_w = int(w * (20 / h))
    
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    top = (28 - new_h) // 2
    bottom = 28 - new_h - top
    left = (28 - new_w) // 2
    right = 28 - new_w - left
    
    mnist_img = cv2.copyMakeBorder(digit_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    
    return mnist_img


input_root = "./custom_mnist_perso_new/"
output_root = "./bdd-mnist/"

for digit_folder in range(10):
    digit_str = str(digit_folder)
    input_dir = os.path.join(input_root, digit_str)
    output_dir = os.path.join(output_root, digit_str)
    
    if not os.path.exists(input_dir):
        print(f"Passage : Le dossier {input_dir} n'existe pas.")
        continue
        
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Traitement du chiffre {digit_str}...")
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                mnist_img = convert_to_mnist(input_path)
                cv2.imwrite(output_path, mnist_img)
            except Exception as e:
                print(f"  Erreur sur {filename} : {e}")

print("Conversion terminée !")
            
            
            