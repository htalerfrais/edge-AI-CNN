"""
Crée les trois dossiers train/val/test à partir de data/mnist_digit/
avec un split stratifié (par classe). Structure ImageFolder conservée.
Dépendance : scikit-learn (pip install scikit-learn)
Exécution : python3 utils/split_dataset.py
"""
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

# Chemins relatifs au dossier du script : on remonte au projet (parent de utils/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "mnist_digit"
OUT_TRAIN = PROJECT_ROOT / "data" / "mnist_digit_train"
OUT_VAL = PROJECT_ROOT / "data" / "mnist_digit_val"
OUT_TEST = PROJECT_ROOT / "data" / "mnist_digit_test"

# Paramètres du split (en dur)
SEED = 0
TRAIN_RATIO = 0.8   # 80 % train
VAL_RATIO = 0.1     # 10 % val, 10 % test


def collect_images_by_class(source_dir):
    """
    Parcourt source_dir (structure 0/, 1/, ..., 9/) et retourne
    listes (paths, labels) pour un split stratifié.
    """
    paths = []
    labels = []
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            label = int(class_dir.name)
        except ValueError:
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in (".bmp", ".png", ".jpg", ".jpeg"):
                paths.append(str(f))
                labels.append(label)

    return paths, labels


def split_indices_stratified(paths, labels, train_ratio, val_ratio, seed):
    """Split stratifié avec sklearn : train_ratio train, val_ratio val, le reste en test."""
    # Premier split : train vs (val+test)
    train_paths, rest_paths, train_labels, rest_labels = train_test_split(
        paths, labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )
    # Deuxième split : val vs test (50/50 du reste pour avoir val_ratio et test_ratio)
    val_ratio_rest = val_ratio / (1 - train_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        rest_paths, rest_labels,
        train_size=val_ratio_rest,
        stratify=rest_labels,
        random_state=seed,
        shuffle=True,
    )
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


def main():
    print("Collecting images from", SOURCE_DIR)
    paths, labels = collect_images_by_class(SOURCE_DIR)
    if not paths:
        print("No images found. Aborting.")
        return 1

    n_total = len(paths)
    n_classes = len(set(labels))
    print(f"Total images: {n_total}, classes: {n_classes}")

    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_indices_stratified(
        paths, labels, TRAIN_RATIO, VAL_RATIO, SEED
    )

    print(f"Split: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")

    # Créer les dossiers et copier
    for out_dir in (OUT_TRAIN, OUT_VAL, OUT_TEST):
        out_dir.mkdir(parents=True, exist_ok=True)
        for c in out_dir.iterdir():
            if c.is_dir():
                shutil.rmtree(c)

    def copy_split(path_list, label_list, out_base):
        out_base = Path(out_base)
        for src_path, label in zip(path_list, label_list):
            class_dir = out_base / str(label)
            class_dir.mkdir(parents=True, exist_ok=True)
            dst_path = class_dir / Path(src_path).name
            shutil.copy2(src_path, dst_path)

    copy_split(train_paths, train_labels, OUT_TRAIN)
    copy_split(val_paths, val_labels, OUT_VAL)
    copy_split(test_paths, test_labels, OUT_TEST)

    print("Done.")
    print(f"  Train: {OUT_TRAIN}")
    print(f"  Val:   {OUT_VAL}")
    print(f"  Test:  {OUT_TEST}")
    return 0


if __name__ == "__main__":
    exit(main())
