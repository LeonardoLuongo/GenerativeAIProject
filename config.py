from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()

MODEL_PATH = ROOT_DIR / "weights"

DATASET_BASIC_PATH = ROOT_DIR / "dataset"
DATASET_CELEBA = DATASET_BASIC_PATH / "celeba"
DATASET_CELEBA_IMGS = DATASET_BASIC_PATH / "img_align_celeba"

TRAIN_RES_PATH = ROOT_DIR / "runs"

OUTPUT_DIR_PATH = ROOT_DIR / "output"