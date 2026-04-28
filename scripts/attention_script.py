import sys
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from pipeline.attention_maps import filter_most_conf_patches, generate_attention_maps

CANCER_MODEL = BASE_DIR / "models" / "CANCER_classification_model.pt"
OUTPUTS_DIR = BASE_DIR / "outputs"


def main():
    
    wsi_path = sys.argv[1]
    model_path = str(CANCER_MODEL)
    patches_path = sys.argv[2]
    results_path = sys.argv[3]
    k = 5
    output_root = str(OUTPUTS_DIR)

    patch_size = (512,512)
    batch_size = 1
    num_workers = 0
    output_level = 0

    top_patches_path = filter_most_conf_patches(patches_csv_path= patches_path,  
                                               slide_results_path=results_path, 
                                               k=k,
                                               output_root=output_root)
    
    figs = generate_attention_maps(CANCER_model_path=model_path, 
                                   wsi_path=wsi_path, 
                                   patches_path=top_patches_path, 
                                   patch_size=patch_size, 
                                   batch_size=batch_size, 
                                   num_workers=num_workers, 
                                   output_level=output_level,
                                   output_root=output_root)

    
    
    output = {"figs_csv":str(figs)}
    print(json.dumps(output))

if __name__ == "__main__":
    main()