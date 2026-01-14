from pathlib import Path

# Questo trova automaticamente la cartella dove si trova QUESTO file (la root)
ROOT_DIR = Path(__file__).parent.resolve()
print(ROOT_DIR)
MODEL_PATH = ROOT_DIR / "models"

DATASET_BASIC_PATH = ROOT_DIR / "dataset"
DATASET_CELEBA = DATASET_BASIC_PATH / "celeba"
DATASET_CELEBA_IMGS = DATASET_CELEBA / "img_align_celeba"
DATASET_CELEBA_LIST_ATTR = DATASET_CELEBA / "list_attr_celeba.txt"
DATASET_CELEBA_PROCESSED_DATA = DATASET_CELEBA / "processed_data"
DATASET_CELEBA_PROCESSED_DATA_64_64 = DATASET_CELEBA_PROCESSED_DATA / "images_64x64"
DATASET_CELEBA_PROCESSED_DATA_LIST_ATTR = DATASET_CELEBA_PROCESSED_DATA / "clean_attributes.txt"
MODEL_CHECKPOINT = ROOT_DIR / "checkpoints"
BETTER_MODEL = MODEL_CHECKPOINT / "ddpm_celebA_ep43.pth"

TRAIN_RES_PATH = ROOT_DIR / "runs"

OUTPUT_DIR_PATH = ROOT_DIR / "output"