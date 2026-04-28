from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
from tiatoolbox.wsicore import WSIReader
import pandas as pd
import gc
from pathlib import Path
from shapely.geometry import Polygon
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

""" TISSUE MASK """
def create_tissue_mask(wsi):
    mask = wsi.tissue_mask(resolution=0.976,units="mpp") # Otsu method is default, Thumbnail level resolution(low)
    return mask



""" PATCH EXTRACTOR """
def create_patch_extractor(wsi,patch_size,resolution,mask=None):
    if mask != None:
        extractor = SlidingWindowPatchExtractor(
                            input_img=wsi,
                            patch_size=patch_size,
                            stride=patch_size,
                            resolution=resolution,  
                            units="mpp",
                            input_mask=mask,)
    else:
        extractor = SlidingWindowPatchExtractor(
                            input_img=wsi,
                            patch_size=patch_size,
                            stride=patch_size,
                            resolution=resolution,  
                            units="mpp",)

    return extractor



""" PATCH SIZE LEVEL 0 """
def get_patch_size_L0(wsi, desired_patch_size:int, desired_resolution:float):

    wsi_info = wsi.info.as_dict()
    patch_um = desired_patch_size * desired_resolution

    wsi_mpp_x, wsi_mpp_y = wsi_info['mpp']
    
    # Convert desired patch size to level 0 pixels for x and y
    patch_size_L0_x = int(round(patch_um/wsi_mpp_x))
    patch_size_L0_y = int(round(patch_um/wsi_mpp_y))

    patch_size_L0 = tuple([patch_size_L0_x,patch_size_L0_y])
    return patch_size_L0

""" GENERATE PATCH ID """

def make_patch_id(file_name, x0, y0, level, patch_size):
    return f"{file_name}_x{x0}_y{y0}_l{level}_p{patch_size}"

""" PATCHING AND SAVING """
def ROI_patching(wsi, specs:dict):

    output_root = Path(specs["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    
    output_file = output_root / "1_patches_224_224_20x.csv"

    
    total_patches = 0
    

    wsi_path = Path(wsi)
    wsi_obj = WSIReader.open(wsi_path)

    # RESOLUTION
    wsi_resolution = wsi_obj.info.mpp

    # PATCH SIZE
    patch_size_L0 = get_patch_size_L0(wsi=wsi_obj, desired_patch_size=specs["desired_patche_size"][0], desired_resolution=specs['desired_resolution'])
    
    # TISSUE MASK
    m = create_tissue_mask(wsi_obj)
    
    # EXTRACTOR
    ex = create_patch_extractor(wsi=wsi_obj, 
                                patch_size = patch_size_L0, 
                                resolution= wsi_resolution, 
                                mask=m)

    
    coords = ex.coordinate_list
    slide_df = pd.DataFrame(coords[:, :2], columns=["x0", "y0"])
    #slide_df['slide_uid'] = slide_uid
    slide_df["patch_size_L0_w"] = patch_size_L0[0]
    slide_df["patch_size_L0_h"] = patch_size_L0[1]
    


    # Generating patch IDs

    file_name = wsi_path.stem
    p_size = patch_size_L0[0]
    level = 0
    slide_df["patch_id"] = (file_name + "_x" + slide_df["x0"].astype(str) + "_y" + slide_df["y0"].astype(str) + "_l" + str(level) + "_p" + str(p_size))

    slide_df = slide_df[[ "patch_id", "x0","y0","patch_size_L0_w","patch_size_L0_h"]]

    slide_df.to_csv(output_file, header=True,index=False)
    
    total_patches += len(slide_df)

    # Return csv file path or just the dataframe??
            
            
    # DELETE temp files

    if wsi_obj is not None:
        del wsi_obj
    if ex is not None:
        del ex
    if m is not None:
        del m

    gc.collect()

    return str(output_file)

""" ROI POSITIVE DATAFRAME"""

def ROI_pos_list(patch_csv_path):

    patches = pd.read_csv(patch_csv_path)
    pos = patches[patches['roi_pred']==1]

    x0 = pos["x0"].values
    y0 = pos["y0"].values
    w = pos["patch_size_L0_w"].values
    h = pos["patch_size_L0_h"].values

    coords = []

    for i in range(len(pos)):
        coords.append([
            (x0[i], y0[i]),
            (x0[i] + w[i], y0[i]),
            (x0[i] + w[i], y0[i] + h[i]),
            (x0[i], y0[i] + h[i])
        ])

    return coords

""" FILTER PATCHES TO ROIs """

def filter_ROIs(coords, patch_csv_path):

    patches = pd.read_csv(patch_csv_path)
    pos = patches[patches['roi_pred'] == 1]

    # Keep only the patches with higher confidence
    pos = pos[pos['roi_conf']>0.7]

    # Capping rois to avoid data explosion
    n_samples = min(1500, len(pos))
    pos = pos.sample(n=n_samples, random_state=42)

    roi_x1 = pos["x0"].to_numpy()
    roi_y1 = pos["y0"].to_numpy()
    roi_x2 = roi_x1 + pos["patch_size_L0_w"].to_numpy()
    roi_y2 = roi_y1 + pos["patch_size_L0_h"].to_numpy()

    coords = np.array(coords)

    px1, py1, px2, py2 = coords[:,0], coords[:,1], coords[:,2], coords[:,3]

    filtered = []

    for i in range(len(coords)):
        px1, py1, px2, py2 = coords[i]

        # intersection box
        inter_x1 = np.maximum(px1, roi_x1)
        inter_y1 = np.maximum(py1, roi_y1)
        inter_x2 = np.minimum(px2, roi_x2)
        inter_y2 = np.minimum(py2, roi_y2)

        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)

        inter_area = inter_w * inter_h
        patch_area = (px2 - px1) * (py2 - py1)

        overlap_ratio = inter_area / patch_area

        if np.any(overlap_ratio >= 0.2): # THRESHOLD
            filtered.append(coords[i])

    return filtered


def CANCER_patching(wsi, specs:dict, patches_csv_path):

    output_root = Path(specs["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    
    output_file = output_root / "3_patches_512_512_40x.csv"

    
    total_patches = 0
    

    wsi_path = Path(wsi)
    wsi_obj = WSIReader.open(wsi_path)

    # RESOLUTION
    wsi_resolution = wsi_obj.info.mpp

    # PATCH SIZE
    patch_size_L0 = specs['desired_patche_size']

    
    # EXTRACTOR
    ex = create_patch_extractor(wsi=wsi_obj, 
                                patch_size = patch_size_L0, 
                                resolution= wsi_resolution,)

    
    coords = ex.coordinate_list

    filtered_coords = filter_ROIs(coords, patches_csv_path)
    filtered_coords = np.array(filtered_coords)

    # convert coordinates to DataFrame, one row per patch
    
    slide_df = pd.DataFrame(filtered_coords[:, :2], columns=["x0", "y0"])
    slide_df["patch_size_L0_w"] = patch_size_L0[0]
    slide_df["patch_size_L0_h"] = patch_size_L0[1]
    

    # Generating patch IDs

    file_name = wsi_path.stem
    p_size = patch_size_L0[0]
    level = 0
    slide_df["patch_id"] = (file_name + "_x" + slide_df["x0"].astype(str) + "_y" + slide_df["y0"].astype(str) + "_l" + str(level) + "_p" + str(p_size))

    slide_df = slide_df[[ "patch_id", "x0","y0","patch_size_L0_w","patch_size_L0_h"]]

    # save to csv
    slide_df.to_csv(output_file, header=True,index=False)
    total_patches += len(slide_df)
            
            
    # DELETE temp files

    if wsi_obj is not None:
        del wsi_obj
    if ex is not None:
        del ex
    

    gc.collect()          

    # Return the string path of the patches csv file
    return str(output_file)  
            

def generate_ROI_thumbnail(wsi_path,patches_path, output_root):
    output_path = Path(output_root) / "wsi_thumbnail_with_selected_rois.png"

    # load image and generate thumbnail photo
    wsi_path = Path(wsi_path)
    wsi = WSIReader.open(wsi_path)
    thumbnail = wsi.slide_thumbnail(resolution=1.25, units="power")

    # scale level 0 patch coordinates to thumbnail level
    wsi_dims = wsi.slide_dimensions(resolution=0, units="level")
    thumb_h, thumb_w = thumbnail.shape[:2]
    scale_x = thumb_w / wsi_dims[0]
    scale_y = thumb_h / wsi_dims[1]

    # load patches .csv
    patches_path = Path(patches_path)
    df = pd.read_csv(patches_path)

    # Generate overlay
    alpha = 0.6

    mask = thumbnail.copy()

    for _, row in df.iterrows():
        x0 = int(row["x0"] * scale_x)
        y0 = int(row["y0"] * scale_y)
        w = int(row["patch_size_L0_w"] * scale_x)
        h = int(row["patch_size_L0_h"] * scale_y)

        cv2.rectangle(mask, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), -1)

    overlay = cv2.addWeighted(mask, alpha, thumbnail, 1 - alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    cv2.imwrite(str(output_path), overlay_rgb)
    
    relative_path = output_path.relative_to(output_root)
    return str(relative_path)


