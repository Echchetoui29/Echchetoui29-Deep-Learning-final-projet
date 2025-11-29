"""
dataset_analysis.py

Ce script permet d'analyser et de visualiser des datasets d'images MRI pour Alzheimer.

Fonctionnalités principales :
1. Charger les datasets originaux et/ou augmentés.
2. Résumer la structure des datasets (nombre d'images, répartition par classe).
3. Visualiser des échantillons d'images par classe.
4. Analyser les statistiques d'images : dimensions, modes, intensité moyenne et écart-type.
5. Comparer datasets originaux et augmentés.
6. Générer des rapports visuels et numériques complets pour chaque dataset ou combinaison.

Auteur : Lamrikhi Abdessamad
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_dataset(data_root: str, dataset_type: str = "original") -> Optional[pd.DataFrame]:
    """
    Charge les images d'un dataset depuis un dossier racine.

    Args:
        data_root (str): Chemin vers le dossier du dataset.
        dataset_type (str): Type du dataset ('original' ou 'augmented').

    Returns:
        pd.DataFrame ou None: DataFrame contenant les chemins d'images et leurs classes.
    """
    if not data_root or not os.path.exists(data_root):
        return None

    data = []
    for class_name in sorted(os.listdir(data_root)):
        class_path = os.path.join(data_root, class_name)
        if os.path.isdir(class_path):
            for img_file in sorted(os.listdir(class_path)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    img_path = os.path.join(class_path, img_file)
                    data.append({
                        'class': class_name,
                        'image_path': img_path,
                        'filename': img_file,
                        'dataset_type': dataset_type
                    })

    return pd.DataFrame(data) if data else None


def load_both_datasets(base_path: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Charge les datasets originaux et augmentés et les combine.

    Args:
        base_path (str): Chemin vers le dossier contenant les datasets.

    Returns:
        Tuple contenant les DataFrames pour le dataset original, augmenté et combiné.
    """
    original_path = os.path.join(base_path, "OriginalDataset")
    augmented_path = os.path.join(base_path, "AugmentedAlzheimerDataset")

    df_original = load_dataset(original_path, "original") if os.path.exists(original_path) else None
    df_augmented = load_dataset(augmented_path, "augmented") if os.path.exists(augmented_path) else None

    df_combined = pd.concat([df_original, df_augmented], ignore_index=True) if df_original is not None and df_augmented is not None else None

    return df_original, df_augmented, df_combined


def print_dataset_summary(df: pd.DataFrame, dataset_type: Optional[str] = None) -> None:
    """
    Affiche un résumé du dataset (nombre d'images et distribution par classe).

    Args:
        df (pd.DataFrame): DataFrame du dataset.
        dataset_type (str, optional): Type du dataset pour le titre.
    """
    title = f"{dataset_type.upper()} DATASET STRUCTURE SUMMARY" if dataset_type else "DATASET STRUCTURE SUMMARY"
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"\nTotal images: {len(df)}")
    print(f"Number of classes: {df['class'].nunique()}\n")
    print("Class distribution:")
    for class_name, count in df['class'].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"   {class_name}: {count} images ({percentage:.1f}%)")
    print()


def visualize_sample_images(df: pd.DataFrame, samples_per_class: int = 3, figsize: Tuple[int, int] = (15, 5), title_suffix: str = "") -> None:
    """
    Affiche des images échantillons par classe.

    Args:
        df (pd.DataFrame): DataFrame du dataset.
        samples_per_class (int): Nombre d'échantillons par classe.
        figsize (Tuple[int, int]): Taille de la figure matplotlib.
        title_suffix (str): Texte à ajouter dans le titre.
    """
    classes = sorted(df['class'].unique())
    num_classes = len(classes)
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(samples_per_class * 5, num_classes * 5))
    if num_classes == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Sample MRI Images by Class{title_suffix}', fontsize=16, fontweight='bold', y=0.995)

    for idx, class_name in enumerate(classes):
        class_images = df[df['class'] == class_name]['image_path'].values
        num_samples = min(samples_per_class, len(class_images))
        sample_indices = np.random.choice(len(class_images), num_samples, replace=False)

        for sample_idx, img_idx in enumerate(sample_indices):
            ax = axes[idx, sample_idx] if num_classes > 1 else axes[sample_idx]
            try:
                img = Image.open(class_images[img_idx])
                ax.imshow(img, cmap='gray')
                ax.set_title(f'{class_name} - Sample {sample_idx+1}', fontsize=10, fontweight='bold')
            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading image\n{str(e)}', ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_class_distribution(df: pd.DataFrame, title_suffix: str = "") -> None:
    """
    Affiche la distribution des classes sous forme de barres et camembert.

    Args:
        df (pd.DataFrame): DataFrame du dataset.
        title_suffix (str): Texte à ajouter dans le titre.
    """
    class_counts = df['class'].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))

  
    bars = axes[0].bar(class_counts.index, class_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Class', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    axes[0].set_title('Class Distribution - Bar Chart', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # Pie chart
    explode = [0.05] * len(class_counts)
    wedges, texts, autotexts = axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', explode=explode, colors=colors, startangle=90, textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    axes[1].set_title('Class Distribution - Pie Chart', fontsize=13, fontweight='bold')

    fig.suptitle(f'Alzheimer\'s MRI Dataset - Class Distribution{title_suffix}', fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / "config" / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    dataset_base_path = os.getenv("DATASET_PATH", "c:/Users/ghout/Desktop/augmented-alzheimer-mri-dataset")
    df_original, df_augmented, df_combined = load_both_datasets(dataset_base_path)

    if df_original is not None and df_augmented is not None:
        original_path = os.path.join(dataset_base_path, "OriginalDataset")
        augmented_path = os.path.join(dataset_base_path, "AugmentedAlzheimerDataset")
       
    elif df_original is not None:
        original_path = os.path.join(dataset_base_path, "OriginalDataset")
       
    elif df_augmented is not None:
        augmented_path = os.path.join(dataset_base_path, "AugmentedAlzheimerDataset")
       
