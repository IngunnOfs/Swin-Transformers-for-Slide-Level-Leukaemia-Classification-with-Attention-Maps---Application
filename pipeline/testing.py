import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))



import torch
import timm
import pandas as pd
from pathlib import Path
from pipeline.datasets import SingleSlideLevelDataset, SingleSlidePatchLevelDataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
import numpy as np

def classify_ROI_patches(ROI_model_path, wsi_path, patches_path, patch_size, batch_size, num_workers, output_level, output_root):


    """ SETTING THE DEVICE """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """ LOADING THE MODEL """
    model = timm.create_model("swin_small_patch4_window7_224", pretrained=False, num_classes=2)

    
    checkpoint = torch.load(ROI_model_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)



    """ LOAD TEST SLIDE """

    wsi_path = Path(wsi_path)
    patches_path = Path(patches_path)

    patches_df = pd.read_csv(patches_path)
    
    test_transform = T.Compose([
                                T.Resize(patch_size),
                                T.ToTensor(),
                                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]) 

    slide_dataset = SingleSlidePatchLevelDataset(patch_df=patches_df,
                                            wsi_path=wsi_path,
                                            transform= test_transform,
                                            output_level= output_level,
                                            patch_size= patch_size)
    test_loader = DataLoader(slide_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers = num_workers)
    
    """ TEST """

    all_preds = []
    all_probs = []
    all_confs = []
    all_patch_ids = []

    model.eval()

    with torch.no_grad():
        for imgs, patch_ids, x0, y0 in test_loader:
            imgs = imgs.to(device)
            

            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:,1]
            preds = (probs >= 0.5).int()
            confs = torch.where(preds == 1, probs, 1 - probs)

            all_patch_ids.extend(patch_ids)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())   

    results_df = pd.DataFrame({
                                'patch_id' : all_patch_ids,
                                'roi_pred': all_preds,
                                'roi_prob' : all_probs,
                                'roi_conf' : all_confs})


    merged_df = patches_df.merge(results_df, on= 'patch_id', how='left')

    output_path = Path(output_root) / "2_patches_with_roi_predictions.csv"

    merged_df.to_csv(output_path, index=False)

    return str(output_path)



def classify_Slide(CANCER_model_path, wsi_path, patches_path, patch_size, batch_size, num_workers, output_level, output_root):
    

    """ SETTING THE DEVICE """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """ LOADING THE MODEL """
    model = timm.create_model("swin_base_patch4_window12_384", pretrained=False, num_classes=4,img_size = patch_size)
    

    
    checkpoint = torch.load(CANCER_model_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)



    """ LOAD TEST SLIDE """

    wsi_path = Path(wsi_path)
    patches_path = Path(patches_path)

    patches_df = pd.read_csv(patches_path)
    
    test_transform = T.Compose([
                                T.Resize(patch_size),
                                T.ToTensor(),
                                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]) 

    slide_dataset = SingleSlideLevelDataset(patch_df=patches_df,
                                            wsi_path=wsi_path,
                                            transform= test_transform,
                                            output_level= output_level,
                                            patch_size= patch_size)
    test_loader = DataLoader(slide_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers = num_workers)
    
    """ TEST """

    all_preds = []
    all_probs = []
    all_confs = []
    all_patch_ids = []
    all_embeddings = []

    chunk_size = 64

    model.eval()

    with torch.no_grad():
        for imgs, patch_ids, x0s, y0s in test_loader:
            patches = imgs.to(device)

            embeddings = []

            for i in range(0, imgs.shape[0], chunk_size):
                patch_chunk = patches[i:i+chunk_size]
                patch_chunk = patch_chunk.to(device)

                emb = model.forward_features(patch_chunk)
                embeddings.append(emb)
            
            patch_embeddings = torch.cat(embeddings, dim= 0)
            all_embeddings.append(patch_embeddings.cpu())


            # PATCH LEVEL

            logits = model.head(patch_embeddings)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            confs = torch.max(probs, dim=1).values

            # store patch level results
            all_patch_ids.extend(patch_ids)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

            

            del patches
            del patch_embeddings
            del logits
    
    all_embeddings = torch.cat(all_embeddings, dim=0).to(device)
    slide_embedding = all_embeddings.mean(dim=0, keepdim=True)
    slide_logits = model.head(slide_embedding)
    slide_prob = torch.softmax(slide_logits, dim=1)
    slide_pred = torch.argmax(slide_prob,dim=1)
    slide_conf = slide_prob[0, slide_pred.item()].item()

               
    # PATCH RESULTS
    results_df = pd.DataFrame({
                                'patch_id' : all_patch_ids,
                                'canc_pred': all_preds,
                                'canc_conf' : all_confs})
    
    probs_array = np.array(all_probs)

    results_df["prob_0"] = probs_array[:, 0]
    results_df["prob_1"] = probs_array[:, 1]
    results_df["prob_2"] = probs_array[:, 2]
    results_df["prob_3"] = probs_array[:, 3]

    
    merged_df = patches_df.merge(results_df, on= 'patch_id', how='left')
    
    
    
    slide_results = pd.DataFrame({
                                    "slide_uid": [wsi_path.stem],
                                    "prediction": [slide_pred.item()],
                                    "confidence": [slide_conf],
                                    "prob_AML": [slide_prob[0, 0].item()],
                                    "prob_ALL": [slide_prob[0, 1].item()],
                                    "prob_CML": [slide_prob[0, 2].item()],
                                    "prob_MPAL": [slide_prob[0, 3].item()]})
    

    output_root = Path(output_root)
    patches_output_path = output_root / "4_patches_with_cancer_predictions.csv"
    slide_output_path = output_root / "4_slide_cancer_prediction.csv"
    
    merged_df.to_csv(patches_output_path, index=False)
    slide_results.to_csv(slide_output_path, index=False)

    return str(patches_output_path), str(slide_output_path)