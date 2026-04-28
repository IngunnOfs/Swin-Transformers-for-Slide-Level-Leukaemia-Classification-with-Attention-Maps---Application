import sys
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from pipeline.testing import classify_ROI_patches

ROI_MODEL = BASE_DIR / "models" / "ROI_classification_model.pt"
OUTPUTS_DIR = BASE_DIR / "outputs"


def main():
    
    wsi_path = sys.argv[1]
    model_path = ROI_MODEL
    patches_path = sys.argv[2]
    patch_size = (224,224)
    batch_size = 16
    num_workers = 0
    output_level = 2
    output_root = str(OUTPUTS_DIR)

    
    output_csv = classify_ROI_patches(model_path, wsi_path, patches_path, patch_size, batch_size, num_workers, output_level, output_root)
    output = {"patches_csv":output_csv}
    print(json.dumps(output))

if __name__ == "__main__":
    main()