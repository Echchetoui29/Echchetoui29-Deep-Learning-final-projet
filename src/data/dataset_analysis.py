"""
dataset_analysis.py

Ce script permet de :
1. Calculer les statistiques de normalisation (moyenne et écart-type) 
   pour un dataset d'images (grayscale actuellement).
2. Calculer les statistiques de dimensions d'images (largeur, hauteur, ratio).
3. Sauvegarder toutes les statistiques dans un fichier JSON.

Le script recherche automatiquement plusieurs variantes de datasets
dans le dossier défini par la variable d'environnement DATASET_PATH.


"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging
from dotenv import load_dotenv
from typing import Dict, Optional
import json

# Chargement des variables d'environnement
env_path = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(dotenv_path=env_path)

logging.basicConfig(level=logging.INFO)


def compute_dataset_statistics(
    data_dir: str, 
    sample_size: Optional[int] = None
) -> Dict[str, float]:
    """
    Calcule la moyenne et l'écart-type des pixels pour la normalisation du dataset.

    Args:
        data_dir (str): Chemin vers le dossier du dataset.
        sample_size (int, optional): Nombre d'images à échantillonner. 
                                     Si None, toutes les images sont utilisées.

    Returns:
        dict: Statistiques comprenant mean, std, total_pixels, num_images, 
              et valeurs correspondantes sur l'échelle 0-255.
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    images = []

    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            class_images = [
                f for f in class_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in valid_extensions
            ]
            images.extend(class_images)
    
    if not images:
        raise ValueError(f"No images found in {data_path}")
    
    print("=" * 70)
    print("COMPUTING DATASET NORMALIZATION STATISTICS")
    print("=" * 70)
    
    if sample_size and len(images) > sample_size:
        np.random.seed(42)
        images = list(np.random.choice(images, sample_size, replace=False))
        print(f"Using sample size: {sample_size}")
    else:
        print(f"Using all images: {len(images)}")
    
    pixel_sum = 0
    pixel_sq_sum = 0
    total_pixels = 0
    
    for img_path in tqdm(images, desc="Processing images"):
        try:
            img = Image.open(img_path).convert('L')
            img_array = np.array(img, dtype=np.float32) / 255.0
            pixel_sum += img_array.sum()
            pixel_sq_sum += (img_array ** 2).sum()
            total_pixels += img_array.size
        except Exception as e:
            logging.warning(f"Error processing {img_path}: {e}")
            continue
    
    mean = pixel_sum / total_pixels
    variance = (pixel_sq_sum / total_pixels) - (mean ** 2)
    std = np.sqrt(variance)
    
    statistics = {
        'mean': float(mean),
        'std': float(std),
        'mean_255': float(mean * 255),
        'std_255': float(std * 255),
        'total_pixels': int(total_pixels),
        'num_images': len(images)
    }
    
    return statistics


def compute_image_dimensions(data_dir: str) -> Dict[str, any]:
    """
    Calcule les statistiques des dimensions des images du dataset.

    Args:
        data_dir (str): Chemin vers le dossier du dataset.

    Returns:
        dict: Statistiques de largeur, hauteur, et ratio d'aspect.
    """
    data_path = Path(data_dir)
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    widths, heights, aspect_ratios = [], [], []

    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    try:
                        img = Image.open(img_path)
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                        aspect_ratios.append(w / h)
                    except Exception as e:
                        logging.warning(f"Error reading {img_path}: {e}")
    
    if not widths:
        return {}

    stats = {
        'width': {
            'min': int(np.min(widths)),
            'max': int(np.max(widths)),
            'mean': float(np.mean(widths)),
            'median': float(np.median(widths)),
            'mode': int(max(set(widths), key=widths.count))
        },
        'height': {
            'min': int(np.min(heights)),
            'max': int(np.max(heights)),
            'mean': float(np.mean(heights)),
            'median': float(np.median(heights)),
            'mode': int(max(set(heights), key=heights.count))
        },
        'aspect_ratio': {
            'min': float(np.min(aspect_ratios)),
            'max': float(np.max(aspect_ratios)),
            'mean': float(np.mean(aspect_ratios))
        }
    }
    
    return stats


def save_statistics(statistics: Dict, output_path: str) -> None:
    """
    Sauvegarde les statistiques dans un fichier JSON.

    Args:
        statistics (dict): Les statistiques à sauvegarder.
        output_path (str): Chemin du fichier JSON de sortie.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"\n✓ Statistics saved to: {output_file}")
