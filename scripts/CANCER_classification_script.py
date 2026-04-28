import sys
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from pipeline.testing import classify_Slide

CANCER_MODEL = BASE_DIR / "models" / "CANCER_classification_model.pt"
OUTPUTS_DIR = BASE_DIR / "outputs"

def main():
    
    wsi_path = sys.argv[1]
    model_path = str(CANCER_MODEL)
    patches_path = sys.argv[2]
    patch_size = (512,512)
    batch_size = 1
    num_workers = 0
    output_level = 0
    output_root = str(OUTPUTS_DIR)

    
    patches_output_csv, slide_output_csv = classify_Slide(model_path, wsi_path, patches_path, patch_size, batch_size, num_workers, output_level, output_root)
    output = {"patch_csv":patches_output_csv, "slide_csv":slide_output_csv}
    print(json.dumps(output))

if __name__ == "__main__":
    main()

