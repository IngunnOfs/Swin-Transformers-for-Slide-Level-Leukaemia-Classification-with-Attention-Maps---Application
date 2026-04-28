import sys
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from pipeline.patching import ROI_patching

OUTPUTS_DIR = BASE_DIR / "outputs"

def main():
    wsi_path = sys.argv[1]

    specs = {"output_root": str(OUTPUTS_DIR),
        "desired_patche_size": (224,224),
        "desired_resolution": 0.5 #0.5 approx 20x
        }
    
    output_csv = ROI_patching(wsi_path, specs)
    output = {"patches_csv":output_csv}
    print(json.dumps(output))

if __name__ == "__main__":
    main()

