from pathlib import Path

# Questo trova automaticamente la cartella dove si trova QUESTO file (la root)
ROOT_DIR = Path(__file__).parent.resolve()
#print(ROOT_DIR)
MODEL_PATH = ROOT_DIR / "models"

DATASET_BASIC_PATH = ROOT_DIR / "dataset"
DATASET_CELEBA = DATASET_BASIC_PATH / "celeba"
DATASET_CELEBA_IMGS = DATASET_BASIC_PATH / "img_align_celeba"

TRAIN_RES_PATH = ROOT_DIR / "runs"

OUTPUT_DIR_PATH = ROOT_DIR / "output"