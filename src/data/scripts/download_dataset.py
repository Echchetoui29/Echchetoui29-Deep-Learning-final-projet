"""
Script pour télécharger et organiser un jeu de données depuis Kaggle.

Ce module utilise kagglehub pour télécharger un dataset, puis le déplace
vers un répertoire de projet spécifié. La configuration est gérée via
des variables d'environnement chargées depuis un fichier .env.
"""
import os
import shutil
import kagglehub
import logging
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(dotenv_path=env_path)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_dataset(dataset_name: str, target_dir: str) -> str:
    """
    Télécharge un dataset depuis Kaggle et le déplace vers un répertoire cible.

    La fonction télécharge le dataset spécifié en utilisant kagglehub dans un
    emplacement temporaire. Ensuite, elle déplace le contenu de cet emplacement
    vers le répertoire cible `target_dir`. Les fichiers ou dossiers existants
    dans `target_dir` sont supprimés avant le déplacement pour garantir une
    copie propre. Le répertoire temporaire est supprimé après l'opération.

    Args:
        dataset_name (str): Le nom du dataset sur Kaggle (format: 'utilisateur/nom-dataset').
        target_dir (str): Le chemin du répertoire où le dataset doit être stocké.

    Raises:
        FileExistsError: Si une erreur de fichier existant se produit lors de l'opération.
        Exception: Pour toute autre erreur liée au téléchargement ou à la manipulation de fichiers.

    Returns:
        str: Le chemin final du répertoire où le dataset a été stocké.
    """
    try:
        logging.info(f"Téléchargement du dataset : {dataset_name}")
        path = kagglehub.dataset_download(dataset_name)
        logging.info(f"Dataset téléchargé dans le répertoire temporaire : {path}")
        
        if target_dir and path != target_dir:
            os.makedirs(target_dir, exist_ok=True)
            
            for item in os.listdir(path):
                src = os.path.join(path, item)
                dst = os.path.join(target_dir, item)
                
                if os.path.exists(dst):
                    logging.warning(f"L'élément existant '{dst}' va être supprimé.")
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                
                shutil.move(src, dst)
            
            logging.info(f"Tous les fichiers ont été déplacés vers : {target_dir}")
            
            shutil.rmtree(path)
            logging.info(f"Le répertoire temporaire '{path}' a été supprimé.")
            path = target_dir
        
        return path

    except FileExistsError as e:
        logging.error(f"Erreur de type 'FileExistsError' : {e}")
        raise
    except Exception as e:
        logging.error(f"Échec du téléchargement ou du traitement du dataset : {e}")
        raise

if __name__ == "__main__":
    dataset_name = os.getenv("KAGGLE_DATASET", "uraninjo/augmented-alzheimer-mri-dataset")
    target_dir = os.getenv("DATASET_PATH", "c:/Users/DELL/Desktop/augmented-alzheimer-mri-dataset")

    try:
        dataset_path = download_dataset(dataset_name, target_dir)
        logging.info(f"Processus terminé. Le dataset est disponible à l'emplacement : {dataset_path}")
    except Exception as e:
        logging.error(f"Une erreur critique est survenue durant l'exécution du script : {e}")