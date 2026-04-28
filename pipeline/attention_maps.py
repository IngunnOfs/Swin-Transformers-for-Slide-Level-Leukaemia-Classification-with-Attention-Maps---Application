import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from pipeline.datasets import SingleSlideLevelDataset

import torch
import timm
import pandas as pd
from pathlib import Path

from torchvision import transforms as T
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from timm.models.swin_transformer import SwinTransformerBlock
import scipy
from scipy.ndimage import gaussian_filter, median_filter

""" ATTENTION HOOK """

def my_block_hook(module, inp, outp):
    """Hook for SwinTransformerBlock that stores attention and shift_size"""
    global global_attention

    x = inp[0]  # x: [B, H, W, C] or [B, C, H, W]
    
    # Ensure channel-last
    if x.shape[1] == module.dim:  # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
    
    B, H, W, C = x.shape
    window_size = module.attn.window_size[0]
    shift_size = module.shift_size[0] if hasattr(module, "shift_size") else 0
    
    attn_module = module.attn
    
    if shift_size > 0:
        x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))

    # Partition into windows
    x_windows = x.unfold(1, window_size, window_size).unfold(2, window_size, window_size)
    # x_windows: [B, nH, nW, window_size, window_size, C]
    x_windows = x_windows.contiguous().view(-1, window_size*window_size, C)  # [B*num_windows, N, C]
    
    qkv = attn_module.qkv(x_windows).reshape(-1, window_size*window_size, 3, attn_module.num_heads, C // attn_module.num_heads)
    qkv = qkv.permute(2,0,3,1,4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    q = q * attn_module.scale
    attn = (q @ k.transpose(-2,-1))
    attn_probs = attn_module.softmax(attn)

    global_attention.append({
        "attn": attn_probs.detach().cpu(),
        "shift_size": shift_size,
        "window_size": window_size
    })

""" CONVERT SWIN WINDOWED ATTENTION TO FULL IMAGE HEATMAP"""
def swin_attn_to_img(attn, window_size, H_feat, W_feat, shift_size):
    # attn : [num_windows, num_patches, num_patches] tensor for one stage
    # window_size : int, number of patches per window
    # H_feat, W_feat : feature map size in patches
   
    # calc average attention over attended patches
    attn_per_patch = attn.mean(dim=2)
    
    # each window is window_size x window_size
    attn_windows = attn_per_patch.reshape(-1, window_size, window_size)
    

    #if shift_size > 0:
        # roll windows back to original position
        #attn_windows = torch.roll(attn_windows, shifts=(-shift_size, -shift_size), dims=(1,2))


    num_windows_h = H_feat // window_size
    num_windows_w = W_feat // window_size

    # arrange windows to full feature map
    rows = []
    for i in range(num_windows_h):
        row_windows = attn_windows[i * num_windows_w:(i+1)*num_windows_w]
        row = torch.cat(list(row_windows), dim = 1)
        rows.append(row)
    img_patch = torch.cat(rows, dim=0)

    # returns image patch as a 2D numpy array, shape [H_feat, W_feat]
    return img_patch.cpu().numpy()

""" GENERATE CUMULATIVE ATTENTION HEATMAP OVER ALL SWIN STAGES"""
def cumulative_swin_attention(global_attention, img_tensor, combine_heads='mean'):
    # global_attention : list of attention tensors from model
    # img_tensor : [C, H, W] torch tensor (0-1)
    # combine heads : mean or max methods

    
    # if the img_tensor is [C,H,W], convert to [H,W,C]
    if img_tensor.shape[0] == 3:
        img_H, img_W = img_tensor.shape[1], img_tensor.shape[2]
    else:
        img_H, img_W = img_tensor.shape[:2]
    
    # generate a blank map to fill
    cumulative_map = np.zeros((img_H, img_W), dtype=np.float32)

    for stage_idx, attn in enumerate(global_attention):
        attn_tensor = attn['attn']
        # attn shape: [num_windows, num_heads, num_patches, num_patches]
        if combine_heads == 'mean':
            attn_stage = attn_tensor.mean(dim=1)  # mean over heads
        elif combine_heads == 'max':
            attn_stage, _ = attn_tensor.max(dim=1)
        else:
            raise ValueError("combine_heads must be 'mean' or 'max'")
        
        num_windows, num_patches, _ = attn_stage.shape
        window_size = int(num_patches**0.5)
        H_feat = int(np.sqrt(num_windows))*window_size
        W_feat = H_feat

        
        
        shift_size = attn['shift_size']
        

        attn_img = swin_attn_to_img(attn_stage, window_size, H_feat, W_feat, shift_size)
        
        # remove padding
        true_tokens = img_H // 4
        attn_img = attn_img[:true_tokens, :true_tokens]

        # upsample to match org. img
        attn_img_upsampled = scipy.ndimage.zoom(attn_img, img_H/attn_img.shape[0])
        attn_min = attn_img_upsampled.min()
        attn_max = attn_img_upsampled.max()
        
        # Normalise each attention map to 0-1 range to ensure it contributes in equal range for each attention layer
        if attn_max - attn_min > 1e-6:
            attn_norm = (attn_img_upsampled - attn_min) / (attn_max - attn_min)
        else:
            attn_norm = attn_img_upsampled

        # add to cumulative map
        cumulative_map += attn_norm
        

    # normalise cumulative map to 0-1 and clipping the outliers with min at 5 and max at 99 (keep the areas with increased attention)
    cumulative_map = (cumulative_map-cumulative_map.min()) / (cumulative_map.max()-cumulative_map.min() + 1e-6)
    low = np.percentile(cumulative_map, 5)
    high = np.percentile(cumulative_map, 99)

    cumulative_map = np.clip(cumulative_map, low, high)

    return cumulative_map

""" SAVE ATTENTION MAPS """

def plot_cumulative_attention(img_plot, cumulative_map, title, output_root):
    
    if img_plot.shape[0] == 3:  # [C,H,W]
        img_np = img_plot.permute(1,2,0).cpu().numpy()
    else:
        img_np = img_plot.cpu().numpy()

    # de-normalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = img_np * std + mean
    img_norm = np.clip(img_norm,0,1)
    
    """ PLOTTING FIGURE """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    
    # Restoring the normalised cumulative map after clipping
    attn_norm = (cumulative_map - cumulative_map.min()) / (
        cumulative_map.max() - cumulative_map.min() + 1e-6
    )
    attn_norm = gaussian_filter(attn_norm, sigma=2)
    attn_norm = median_filter(attn_norm, size=3)
    
    
    # --- Create overlay ---
    heatmap_rgb = plt.cm.jet(attn_norm)[:, :, :3] 
    overlay = 0.7 * img_norm + 0.3 * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)
    
    # Panel 1 — Original
    axes[0].imshow(img_norm)
    axes[0].set_title("Original")

    
    # Panel 2 — Overlay
    axes[1].imshow(overlay)
    axes[1].set_title("Cumulative Attention Overlay")

    
    # Panel 3 — Pure Heatmap
    axes[2].imshow(attn_norm, cmap="jet")
    axes[2].set_title("Cumulative Attention Heatmap")

    
    plt.tight_layout()

    output_parent = Path(output_root) / "Attention_Images"
    output_parent.mkdir(parents=True, exist_ok=True)
    output_path = output_parent / title


    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    relative_path = output_path.relative_to(output_root)

    return str(relative_path)

def generate_attention_maps(CANCER_model_path, wsi_path, patches_path, patch_size, batch_size, num_workers, output_level, output_root):
    class_map = {0 : "AML", 1: "ALL", 2: "CML", 3: "MPAL"}

    """ SETTING THE DEVICE """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """ LOADING THE MODEL """
    model = timm.create_model("swin_base_patch4_window12_384", pretrained=False, num_classes=4,img_size = patch_size)

    
    checkpoint = torch.load(CANCER_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
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
    
    """ ATTENTION """
    
    

    model.eval()

    attention_rows = []

    """ ATTACHING THE ATTENTION HOOK"""
    for name, module in model.named_modules():
        if isinstance(module, SwinTransformerBlock):
            module.register_forward_hook(my_block_hook)

    global global_attention
    global_attention = []

    
    for imgs, patch_ids, x0s, y0s in test_loader:
        imgs = imgs.to(device)

        for i in range(imgs.shape[0]):
            img = imgs[i].unsqueeze(0)
            patch_id = patch_ids[i]

            global_attention = []

            img_cpu = img.squeeze(0).cpu()
            img_plot = np.transpose(img_cpu,(1,2,0))


            with torch.no_grad():
                logits = model(img)

        title = f"Cumulative_attention_patch_{patch_id}.png"

        last_attention = global_attention[-24:] if len(global_attention) > 24 else global_attention

        cumulative_map = cumulative_swin_attention(last_attention, img_plot, combine_heads='max')
        fig_path = plot_cumulative_attention(img_plot, cumulative_map, title,  output_root=output_root)
        
        attention_rows.append({"path_id": patch_id, "fig_path":fig_path})

        
    results_df = pd.DataFrame(attention_rows)
    output_path = Path(output_root) / "5_attention_results.csv"

    results_df.to_csv(output_path, index=False)
    
    return str(output_path)

        
def filter_most_conf_patches(patches_csv_path, slide_results_path, k, output_root):
    


    patches_csv_path = Path(patches_csv_path)
    slide_results_path = Path(slide_results_path)

    patches_df = pd.read_csv(patches_csv_path)
    slide_results_df = pd.read_csv(slide_results_path)

    slide_label = slide_results_df["prediction"].iloc[0]
    column_name = str("prob_" + str(slide_label))

    top_patches = (patches_df.sort_values(by=column_name, ascending=False).head(k))

    output_path = Path(output_root) / "5_top_patches.csv"
    top_patches.to_csv(output_path, index=False)

    return output_path





