import sys
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from pipeline.patching import CANCER_patching, generate_ROI_thumbnail

OUTPUTS_DIR = BASE_DIR / "outputs"

def main():
    wsi_path = sys.argv[1]
    patches_csv_path = sys.argv[2]

    specs = {"output_root": str(OUTPUTS_DIR),
        "desired_patche_size": (512,512),
        "desired_resolution": 0 
        }
    
    output_csv = CANCER_patching(wsi_path, specs, patches_csv_path)
    output_png = generate_ROI_thumbnail(wsi_path, output_csv, OUTPUTS_DIR)
    output = {"patches_csv":output_csv, "wsi_thumbnail":output_png}
    print(json.dumps(output))

if __name__ == "__main__":
    main()

