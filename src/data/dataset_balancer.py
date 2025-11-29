"""
dataset_balancer.py

Ce script permet de :
1. Équilibrer un dataset d'images par duplication et rotation.
2. Générer un dataset équilibré à partir des datasets originaux, augmentés ou combinés.
3. Fournir un rapport détaillé de l'équilibrage et des visualisations de son impact.

Fonctionnalités principales :
- Classe DatasetBalancer : gestion de l'équilibrage des classes par augmentation.
- Création de datasets équilibrés depuis un dataset combiné.
- Analyse et visualisation de l'impact de l'équilibrage.
- Détection automatique des datasets disponibles dans le répertoire de base.


"""

import os
import shutil
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class DatasetBalancer:
    """
    Classe pour équilibrer un dataset d'images.

    Paramètres :
    ------------
    input_dir : str
        Chemin vers le dataset d'origine.
    output_dir : str
        Chemin vers le dataset équilibré généré.
    target_size : int, optionnel
        Nombre cible d'images par classe. Si None, utilise la taille de la classe la plus grande.
    """

    def __init__(self, input_dir: str, output_dir: str, target_size: Optional[int] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size

        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_distribution = self._get_class_distribution()

        if self.target_size is None:
            self.target_size = max(self.class_distribution.values())

    def _get_class_distribution(self) -> Dict[str, int]:
        """
        Retourne la distribution initiale des classes (nombre d'images par classe).
        """
        distribution = {}
        for class_dir in sorted(self.input_dir.iterdir()):
            if class_dir.is_dir():
                images = [
                    f for f in class_dir.iterdir()
                    if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
                ]
                distribution[class_dir.name] = len(images)
        return distribution

    def _rotate_image(self, image_path: Path, angle: float) -> Image.Image:
        """
        Retourne une image pivotée de l'angle spécifié.
        """
        img = Image.open(image_path)

        if img.mode in ['L', 'P']:
            img = img.convert('L')
        elif img.mode not in ['RGB', 'RGBA']:
            img = img.convert('RGB')

        rotated = img.rotate(angle, expand=False, fillcolor='white' if img.mode == 'RGB' else 255)
        return rotated

    def _duplicate_image(self, image_path: Path) -> Image.Image:
        """
        Retourne une copie de l'image.
        """
        img = Image.open(image_path)

        if img.mode in ['L', 'P']:
            img = img.convert('L')
        elif img.mode not in ['RGB', 'RGBA']:
            img = img.convert('RGB')

        return img

    def _generate_augmented_samples(self, class_name: str, num_samples: int) -> List[Tuple[Image.Image, str]]:
        """
        Génère des images supplémentaires pour équilibrer une classe.
        """
        class_path = self.input_dir / class_name
        original_images = sorted([
            f for f in class_path.iterdir()
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        ])

        augmented_samples = []
        rotation_angles = [5, 10, 15, 20, 25, 30, -5, -10, -15, -20, -25, -30]

        current_count = 0
        rotation_idx = 0
        img_idx = 0

        while current_count < num_samples:
            original_img = original_images[img_idx % len(original_images)]

            if current_count % 3 == 0:
                img = self._duplicate_image(original_img)
                filename = f"{original_img.stem}_dup_{current_count}{original_img.suffix}"
            else:
                angle = rotation_angles[rotation_idx % len(rotation_angles)]
                img = self._rotate_image(original_img, angle)
                filename = f"{original_img.stem}_rot_{angle}_{current_count}{original_img.suffix}"
                rotation_idx += 1

            augmented_samples.append((img, filename))
            current_count += 1
            img_idx += 1

        return augmented_samples

    def balance_dataset(self, strategy: str = "rotation_duplication") -> Dict[str, int]:
        """
        Équilibre le dataset en copiant et en augmentant les images pour atteindre la taille cible.

        Retour :
        --------
        final_distribution : dict
            Distribution finale des classes après équilibrage.
        """
        print("\n" + "="*70)
        print(f"BALANCING DATASET ({strategy.upper()})")
        print("="*70)

        final_distribution = {}

        for class_name in sorted(self.class_distribution.keys()):
            class_input_path = self.input_dir / class_name
            class_output_path = self.output_dir / class_name
            class_output_path.mkdir(parents=True, exist_ok=True)

            current_count = self.class_distribution[class_name]
            gap = self.target_size - current_count

            print(f"Processing {class_name}: {current_count} -> {self.target_size}")

            original_images = sorted([
                f for f in class_input_path.iterdir()
                if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
            ])

            for img_file in tqdm(original_images, desc=f"{class_name}", leave=False):
                shutil.copy2(img_file, class_output_path / img_file.name)

            if gap > 0:
                augmented_samples = self._generate_augmented_samples(class_name, gap)
                for img, filename in tqdm(augmented_samples, desc=f"{class_name} Aug", leave=False):
                    output_path = class_output_path / filename
                    img.save(output_path)

            final_count = len([
                f for f in class_output_path.iterdir()
                if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
            ])

            final_distribution[class_name] = final_count

        return final_distribution

    def generate_report(self, final_distribution: Optional[Dict[str, int]] = None) -> None:
        """
        Génère un rapport textuel des effets de l'équilibrage.
        """
        if final_distribution is None:
            final_distribution = self._get_class_distribution_from_path(self.output_dir)

        print("\n" + "="*70)
        print("BALANCING REPORT")
        print("="*70)
        print(f"\n{'Class':<20} | {'Before':>8} | {'After':>8} | {'Added':>8}")
        print("-" * 70)

        total_before = 0
        total_after = 0
        total_added = 0

        for class_name in sorted(self.class_distribution.keys()):
            before = self.class_distribution[class_name]
            after = final_distribution.get(class_name, 0)
            added = after - before

            total_before += before
            total_after += after
            total_added += added

            print(f"{class_name:<20} | {before:>8} | {after:>8} | {added:>8}")

        print("-" * 70)
        print(f"{'TOTAL':<20} | {total_before:>8} | {total_after:>8} | {total_added:>8}")
        print("-" * 70 + "\n")

    @staticmethod
    def _get_class_distribution_from_path(directory: Path) -> Dict[str, int]:
        """
        Retourne la distribution des classes à partir d'un répertoire donné.
        """
        distribution = {}
        for class_dir in sorted(directory.iterdir()):
            if class_dir.is_dir():
                images = [
                    f for f in class_dir.iterdir()
                    if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
                ]
                distribution[class_dir.name] = len(images)
        return distribution


def create_balanced_dataset_from_combined(
    combined_dir: str,
    output_dir: str,
    target_size: Optional[int] = None
) -> DatasetBalancer:
    """
    Crée un dataset équilibré à partir d'un dataset combiné en appliquant duplication et rotation.
    """
    balancer = DatasetBalancer(combined_dir, output_dir, target_size)
    final_distribution = balancer.balance_dataset(strategy="rotation_duplication")
    balancer.generate_report(final_distribution)
    return balancer


def analyze_balancing_impact(before_dir: str, after_dir: str) -> pd.DataFrame:
    """
    Analyse l'impact de l'équilibrage sur le dataset et retourne un DataFrame avec les stats.
    """
    before_dist = DatasetBalancer._get_class_distribution_from_path(Path(before_dir))
    after_dist = DatasetBalancer._get_class_distribution_from_path(Path(after_dir))

    analysis = []
    for class_name in sorted(before_dist.keys()):
        before_count = before_dist.get(class_name, 0)
        after_count = after_dist.get(class_name, 0)
        added = after_count - before_count
        pct_change = (added / before_count * 100) if before_count > 0 else 0

        analysis.append({
            'Class': class_name,
            'Before': before_count,
            'After': after_count,
            'Added': added,
            'Percentage_Change': pct_change,
            'Balance_Ratio': after_count / before_count if before_count > 0 else 1.0
        })

    return pd.DataFrame(analysis)


def visualize_balancing_impact(before_dir: str, after_dir: str) -> None:
    """
    Génère des graphiques pour visualiser l'impact de l'équilibrage du dataset.
    """
    import matplotlib.pyplot as plt

    analysis_df = analyze_balancing_impact(before_dir, after_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Graphique Before vs After
    ax = axes[0, 0]
    x = np.arange(len(analysis_df))
    width = 0.35
    bars1 = ax.bar(x - width/2, analysis_df['Before'], width, label='Before', color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x + width/2, analysis_df['After'], width, label='After', color='#27ae60', edgecolor='black')
    ax.set_xlabel('Class', fontweight='bold')
    ax.set_ylabel('Number of Images', fontweight='bold')
    ax.set_title('Dataset Balancing - Before vs After', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(analysis_df['Class'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Graphique Images Added
    ax = axes[0, 1]
    colors = plt.cm.Set3(np.linspace(0, 1, len(analysis_df)))
    bars = ax.bar(analysis_df['Class'], analysis_df['Added'], color=colors, edgecolor='black')
    ax.set_xlabel('Class', fontweight='bold')
    ax.set_ylabel('Number of Images Added', fontweight='bold')
    ax.set_title('Images Added per Class', fontweight='bold', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # Graphique Percentage Change
    ax = axes[1, 0]
    colors_pct = ['#27ae60' if x >= 0 else '#e74c3c' for x in analysis_df['Percentage_Change']]
    bars = ax.bar(analysis_df['Class'], analysis_df['Percentage_Change'], color=colors_pct, edgecolor='black')
    ax.set_xlabel('Class', fontweight='bold')
    ax.set_ylabel('Percentage Change (%)', fontweight='bold')
    ax.set_title('Percentage Change in Class Size', fontweight='bold', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')

    # Graphique Pie Chart
    ax = axes[1, 1]
    labels = analysis_df['Class']
    after_values = analysis_df['After']
    colors = plt.cm.Set3(np.linspace(0, 1, len(analysis_df)))
    wedges, texts, autotexts = ax.pie(after_values, labels=labels, autopct='%1.1f%%',
                                      colors=colors, startangle=90)
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    ax.set_title('Balanced Dataset Distribution', fontweight='bold', fontsize=12)

    fig.suptitle('Dataset Balancing Impact Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def balance_available_datasets(dataset_base: str) -> None:
    """
    Détecte les datasets disponibles et applique l'équilibrage.
    """
    original_path = os.path.join(dataset_base, "OriginalDataset")
    augmented_path = os.path.join(dataset_base, "AugmentedAlzheimerDataset")
    combined_path = os.path.join(dataset_base, "CombinedAlzheimerDataset")

    has_original = os.path.exists(original_path)
    has_augmented = os.path.exists(augmented_path)
    has_combined = os.path.exists(combined_path)

    print("\nDataset Detection:")
    print(f"  Original: {'Found' if has_original else 'Not Found'}")
    print(f"  Augmented: {'Found' if has_augmented else 'Not Found'}")
    print(f"  Combined: {'Found' if has_combined else 'Not Found'}")

    input_path = None
    input_name = None

    if has_combined:
        input_path = combined_path
        input_name = "CombinedAlzheimerDataset"
        output_name = "BalancedFromCombined"
    elif has_augmented:
        input_path = augmented_path
        input_name = "AugmentedAlzheimerDataset"
        output_name = "BalancedFromAugmented"
    elif has_original:
        input_path = original_path
        input_name = "OriginalDataset"
        output_name = "BalancedFromOriginal"

    if input_path is None:
        print("\nError: No dataset found\n")
        return

    output_path = os.path.join(dataset_base, f"Balanced{output_name}")
    print(f"\nBalancing: {input_name} -> Balanced{output_name}\n")

    balancer = create_balanced_dataset_from_combined(input_path, output_path, target_size=12800)

    try:
        visualize_balancing_impact(input_path, output_path)
    except Exception:
        pass


if __name__ == "__main__":
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / "config" / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    DATASET_BASE = os.getenv("DATASET_PATH", "c:/Users/ghout/Desktop/augmented-alzheimer-mri-dataset")
    balance_available_datasets(DATASET_BASE)
