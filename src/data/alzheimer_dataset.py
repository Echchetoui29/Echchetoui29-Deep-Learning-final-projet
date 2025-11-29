"""
AlzheimerMRIDataset.py

Ce script définit une classe PyTorch Dataset pour la classification
des images MRI liées à la maladie d'Alzheimer. Il permet de :

1. Charger automatiquement les images depuis des sous-répertoires de classes.
2. Appliquer un redimensionnement des images à une taille cible.
3. Appliquer des transformations optionnelles sur les images.
4. Fournir un accès simple aux images et labels via le protocole Dataset PyTorch.
5. Récupérer la distribution des classes dans le dataset.
6. Obtenir le nom d'une classe à partir de son index.

Le script inclut également un bloc __main__ pour tester le chargement
du dataset et vérifier la distribution des classes.

Auteur : Lamrikhi Abdessamad
"""
import os
from pathlib import Path
from typing import Tuple, Optional, Callable
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AlzheimerMRIDataset(Dataset):
    """
    PyTorch Dataset pour la classification des images MRI Alzheimer.

    Cette classe charge les images à partir de sous-répertoires de classes,
    applique un redimensionnement et des transformations optionnelles.

    Attributs de classe:
        CLASS_NAMES: Noms standards des classes.
        CLASS_TO_IDX: Mapping nom de classe -> index.
        IDX_TO_CLASS: Mapping index -> nom de classe.
    """

    CLASS_NAMES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASS_NAMES)}
    IDX_TO_CLASS = {idx: cls_name for cls_name, idx in CLASS_TO_IDX.items()}

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (200, 190)
    ):
        """
        Initialisation du dataset.

        Args:
            root_dir: Répertoire racine contenant les sous-dossiers de classes.
            transform: Transformations optionnelles à appliquer sur les images.
            target_size: Taille cible des images (hauteur, largeur).
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_size = target_size

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Répertoire dataset introuvable : {root_dir}")

        self.images = []
        self.labels = []

        self._load_images()

    def _load_images(self) -> None:
        """
        Charge les chemins d'images et leurs labels depuis la structure de répertoires.
        """
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])

        if not class_dirs:
            raise ValueError(f"Aucun sous-dossier de classes trouvé dans {self.root_dir}")

        for class_dir in class_dirs:
            class_name = class_dir.name

            if class_name not in self.CLASS_TO_IDX:
                logger.warning(f"Classe inconnue : {class_name}. Ignorée.")
                continue

            class_idx = self.CLASS_TO_IDX[class_name]

            image_files = [
                f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in valid_extensions
            ]

            logger.info(f"{len(image_files)} images trouvées dans la classe {class_name}")

            for img_path in image_files:
                self.images.append(str(img_path))
                self.labels.append(class_idx)

        logger.info(f"Nombre total d'images chargées : {len(self.images)}")

    def __len__(self) -> int:
        """Retourne la taille totale du dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Récupère une image et son label.

        Args:
            idx: Index de l'échantillon.

        Returns:
            Tuple (image, label) où image est un tenseur PyTorch.
        """
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('L')
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)

            if self.transform:
                image = self.transform(image)
            else:
                image = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).unsqueeze(0)

            return image, label

        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'image {img_path} : {e}")
            return torch.zeros(1, *self.target_size), label

    def get_class_name(self, idx: int) -> str:
        """
        Retourne le nom de classe correspondant à un index.

        Args:
            idx: Index de la classe.

        Returns:
            Nom de la classe.
        """
        return self.IDX_TO_CLASS[idx]

    def get_class_distribution(self) -> dict:
        """
        Retourne la distribution des classes dans le dataset.

        Returns:
            Dictionnaire {nom_classe: nombre_d'images}.
        """
        distribution = {}
        for label in self.labels:
            class_name = self.IDX_TO_CLASS[label]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution


if __name__ == "__main__":
    import logging
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)

    env_path = Path(__file__).parent.parent / "config" / ".env"
    load_dotenv(dotenv_path=env_path)

    DATASET_BASE = os.getenv("DATASET_PATH", "c:/Users/ghout/Desktop/augmented-alzheimer-mri-dataset")

    datasets_to_check = [
        ("BalancedAlzheimerDataset", "Balanced"),
        ("CombinedAlzheimerDataset", "Combined"),
    ]

    dataset_path = None
    for dataset_dir, name in datasets_to_check:
        full_path = os.path.join(DATASET_BASE, dataset_dir)
        if os.path.exists(full_path):
            dataset_path = full_path
            print(f"✓ Dataset trouvé : {name}")
            break

    if dataset_path:
        dataset = AlzheimerMRIDataset(root_dir=dataset_path)
        print(f"\nTaille du dataset : {len(dataset)}")
        print(f"Distribution des classes : {dataset.get_class_distribution()}")

        image, label = dataset[0]
        print(f"\nShape de l'image exemple : {image.shape}")
        print(f"Label exemple : {label} ({dataset.get_class_name(label)})")
