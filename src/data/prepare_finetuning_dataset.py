"""
Utilitaire de préparation pour un jeu de données d'images réalistes.

Ce script organise un ensemble d'images dans une structure compatible avec l'entraînement
ou le fine-tuning d’un modèle. Il peut également calculer des statistiques globales
sur le jeu de données.

Structure attendue :
    realistic_dataset/
        NonDemented/
        VeryMildDemented/
        MildDemented/
        ModerateDemented/

Utilisation :
    python preparation_dataset_realiste.py --input_dir /chemin/vers/images \
                                           --output_dir /chemin/vers/dataset \
                                           --compute_stats
"""

import os
import argparse
import shutil
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CLASS_NAMES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
ALTERNATIVE_NAMES = {
    'non_demented': 'NonDemented',
    'nondemented': 'NonDemented',
    'non': 'NonDemented',
    'normal': 'NonDemented',
    'very_mild_demented': 'VeryMildDemented',
    'verymilddemented': 'VeryMildDemented',
    'very_mild': 'VeryMildDemented',
    'mild_demented': 'MildDemented',
    'milddemented': 'MildDemented',
    'mild': 'MildDemented',
    'moderate_demented': 'ModerateDemented',
    'moderatedemented': 'ModerateDemented',
    'moderate': 'ModerateDemented'
}


def organize_dataset(
    input_dir: str, 
    output_dir: str, 
    copy_files: bool = True,
    max_total_samples: Optional[int] = None,
    balanced: bool = True
) -> Dict[str, int]:
    """
    Organise un jeu de données dans la structure standard et applique un échantillonnage optionnel.

    Args:
        input_dir: Répertoire source contenant les images.
        output_dir: Répertoire de sortie organisé.
        copy_files: Copier (True) ou déplacer (False) les fichiers.
        max_total_samples: Nombre total maximal d’images à inclure.
        balanced: Répartition équilibrée entre les classes si True.

    Returns:
        Un dictionnaire contenant le nombre final d’images par classe.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Dossier d’entrée introuvable : {input_dir}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    for class_name in CLASS_NAMES:
        (output_path / class_name).mkdir(exist_ok=True)
    
    logger.info(f"Organisation du dataset depuis {input_dir} vers {output_dir}")
    logger.info(f"Mode : {'copie' if copy_files else 'déplacement'}")
    if max_total_samples:
        logger.info(f"Échantillonnage : {max_total_samples} images au total ({'équilibré' if balanced else 'proportionnel'})")
    
    counts = {class_name: 0 for class_name in CLASS_NAMES}
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    subdirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    if subdirs:
        logger.info("Répertoires détectés : traitement par classe…")
        
        class_images = {}
        for subdir in subdirs:
            subdir_name = subdir.name.lower().replace(' ', '_')
            
            if subdir_name in ALTERNATIVE_NAMES:
                class_name = ALTERNATIVE_NAMES[subdir_name]
            elif subdir.name in CLASS_NAMES:
                class_name = subdir.name
            else:
                logger.warning(f"Classe inconnue : {subdir.name}, ignorée.")
                continue
            
            image_files = [f for f in subdir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
            class_images[class_name] = image_files
            logger.info(f"{len(image_files)} images trouvées dans {subdir.name} → {class_name}")
        
        samples_per_class = {}
        if max_total_samples and len(class_images) > 0:
            if balanced:
                base_samples = max_total_samples // len(class_images)
                for class_name in class_images:
                    samples_per_class[class_name] = min(base_samples, len(class_images[class_name]))
                logger.info(f"Échantillonnage équilibré : ~{base_samples} par classe")
            else:
                total_available = sum(len(imgs) for imgs in class_images.values())
                for class_name, images in class_images.items():
                    proportion = len(images) / total_available
                    samples_per_class[class_name] = min(
                        int(max_total_samples * proportion),
                        len(images)
                    )
                logger.info("Échantillonnage proportionnel à la distribution initiale")
        else:
            for class_name, images in class_images.items():
                samples_per_class[class_name] = len(images)
        
        logger.info("Plan d’échantillonnage :")
        for class_name, n_samples in samples_per_class.items():
            available = len(class_images.get(class_name, []))
            logger.info(f"  {class_name}: {n_samples}/{available}")
        logger.info(f"  Total prévu : {sum(samples_per_class.values())} images")
        
        import random
        random.seed(42)
        
        for class_name, image_files in class_images.items():
            n_samples = samples_per_class[class_name]
            selected_files = random.sample(image_files, n_samples) if n_samples < len(image_files) else image_files
            
            logger.info(f"Traitement {class_name}: {len(selected_files)} images")
            
            for img_file in tqdm(selected_files, desc=f"Copie {class_name}"):
                dest = output_path / class_name / img_file.name
                if dest.exists():
                    base = dest.stem
                    ext = dest.suffix
                    counter = 1
                    while dest.exists():
                        dest = output_path / class_name / f"{base}_{counter}{ext}"
                        counter += 1
                
                if copy_files:
                    shutil.copy2(img_file, dest)
                else:
                    shutil.move(str(img_file), dest)
                
                counts[class_name] += 1
    else:
        logger.warning("Aucune sous-dossier détecté. Veuillez organiser les images manuellement.")
        return counts
    
    logger.info("Organisation terminée.")
    for class_name, count in counts.items():
        logger.info(f"{class_name}: {count} images")
    
    return counts


def compute_dataset_statistics(dataset_dir: str, sample_size: int = 1000) -> Dict:
    """
    Calcule la moyenne et l’écart-type des pixels pour la normalisation.

    Args:
        dataset_dir: Répertoire contenant le dataset organisé.
        sample_size: Nombre maximal d’images utilisées pour le calcul.

    Returns:
        Un dictionnaire contenant les statistiques globales.
    """
    logger.info(f"Calcul des statistiques du dataset : {dataset_dir}")
    
    dataset_path = Path(dataset_dir)
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    image_paths = []
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            images = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
            image_paths.extend(images)
    
    logger.info(f"{len(image_paths)} images détectées.")
    
    if len(image_paths) > sample_size:
        logger.info(f"Échantillonnage : {sample_size} images.")
        import random
        random.seed(42)
        image_paths = random.sample(image_paths, sample_size)
    
    pixel_values = []
    sizes = []
    
    transform = transforms.Compose([
        transforms.Resize((200, 190)),
        transforms.ToTensor()
    ])
    
    logger.info("Traitement des images…")
    for img_path in tqdm(image_paths):
        try:
            image = Image.open(img_path).convert('L')
            sizes.append(image.size)
            tensor = transform(image)
            pixel_values.append(tensor.numpy())
        except Exception as e:
            logger.warning(f"Erreur lors du traitement {img_path}: {e}")
            continue
    
    all_pixels = np.concatenate([arr.flatten() for arr in pixel_values])
    mean = float(np.mean(all_pixels))
    std = float(np.std(all_pixels))
    
    size_counts = {}
    for size in sizes:
        size_counts[size] = size_counts.get(size, 0) + 1
    
    most_common_size = max(size_counts.items(), key=lambda x: x[1])[0]
    
    stats = {
        'normalisation': {
            'moyenne': mean,
            'ecart_type': std
        },
        'infos_dataset': {
            'nombre_images': len(image_paths),
            'taille_la_plus_courante': most_common_size,
            'nombre_tailles_uniques': len(size_counts)
        }
    }
    
    logger.info(f"Moyenne : {mean:.6f} — Écart-type : {std:.6f}")
    logger.info(f"Taille la plus fréquente : {most_common_size}")
    
    return stats


def validate_dataset_structure(dataset_dir: str) -> bool:
    """
    Vérifie que le dataset respecte bien la structure attendue.

    Args:
        dataset_dir: Répertoire racine du dataset.

    Returns:
        True si la structure est correcte, False sinon.
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        logger.error(f"Répertoire introuvable : {dataset_dir}")
        return False
    
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    all_valid = True
    
    logger.info(f"Validation de la structure du dataset : {dataset_dir}")
    
    for class_name in CLASS_NAMES:
        class_dir = dataset_path / class_name
        
        if not class_dir.exists():
            logger.error(f"Dossier manquant : {class_name}")
            all_valid = False
            continue
        
        images = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
        
        if len(images) == 0:
            logger.warning(f"Aucune image trouvée dans {class_name}")
            all_valid = False
        else:
            logger.info(f"{class_name}: {len(images)} images")
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(description='Préparation d’un dataset réaliste pour le fine-tuning')
    parser.add_argument('--input_dir', type=str, required=True, help='Dossier contenant les images source')
    parser.add_argument('--output_dir', type=str, required=True, help='Dossier de sortie')
    parser.add_argument('--max_samples', type=int, default=10000, help='Nombre maximal d’images (défaut : 10000)')
    parser.add_argument('--no_sampling', action='store_true', help='Aucun échantillonnage, utiliser toutes les images')
    parser.add_argument('--proportional', action='store_true', help='Échantillonnage proportionnel')
    parser.add_argument('--compute_stats', action='store_true', help='Calculer les statistiques du dataset')
    parser.add_argument('--validate_only', action='store_true', help='Seulement valider la structure')
    parser.add_argument('--move', action='store_true', help='Déplacer au lieu de copier')
    parser.add_argument('--stats_output', type=str, default='stats_dataset_reel.json', help='Fichier de sortie pour les statistiques')
    
    args = parser.parse_args()
    
    if args.validate_only:
        is_valid = validate_dataset_structure(args.input_dir)
        if is_valid:
            logger.info("Structure valide ✔")
        else:
            logger.error("Structure invalide ✗")
        return
    
    counts = organize_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        copy_files=not args.move,
        max_total_samples=None if args.no_sampling else args.max_samples,
        balanced=not args.proportional
    )
    
    if not validate_dataset_structure(args.output_dir):
        logger.error("Erreur : structure finale invalide.")
        return
    
    if args.compute_stats:
        stats = compute_dataset_statistics(args.output_dir)
        stats_path = Path(args.stats_output)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistiques enregistrées dans : {stats_path}")
    
    logger.info("Préparation du dataset terminée.")
    logger.info(f"Dossier final : {args.output_dir}")


if __name__ == "__main__":
    main()
