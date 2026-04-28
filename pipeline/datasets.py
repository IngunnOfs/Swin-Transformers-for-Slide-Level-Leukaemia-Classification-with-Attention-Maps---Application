""" IMPORTING LIBRARIES AND INHERETANCE CLASS """
from torch.utils.data import Dataset

import openslide


""" SLIDE DATASET (testing)"""

class SingleSlidePatchLevelDataset(Dataset):
    
    def __init__(self, patch_df, wsi_path, transform = None, output_level = 0,patch_size = (224,224)):
        self.df = patch_df.reset_index(drop=True)
        self.wsi_path = str(wsi_path)
        self.transform = transform
        self.output_level = output_level
        self.patch_size = patch_size
        
        self.slide = openslide.OpenSlide(self.wsi_path)
        
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
    
        row = self.df.iloc[idx]
        
        # retrieve information
        patch_id = str(row['patch_id'])
        x0 = int(row["x0"])
        y0 = int(row["y0"])
        
        

    
        # read patch from desired output level, default 0
        img = self.slide.read_region(location=(x0, y0),  # level-0 coords
                                level=self.output_level,
                                size=self.patch_size).convert("RGB")

        # apply transformations:
        if self.transform:
            img = self.transform(img)
        
        
        return img, patch_id,x0,y0


    def __del__(self):
        if hasattr(self, "slide"):
            self.slide.close()



class SingleSlideLevelDataset(Dataset):
    # class_map = {"AML": 0, "ALL": 1, "CML": 2, "MPAL": 3}
    def __init__(self, patch_df, wsi_path, transform = None, output_level = 0,patch_size = (512,512)):
        
        self.df = patch_df.reset_index(drop=True)
        self.wsi_path = str(wsi_path)
        self.transform = transform
        self.output_level = output_level
        self.patch_size = patch_size

        self.slide = openslide.OpenSlide(self.wsi_path)


        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        patch_id = str(row["patch_id"])
        x0 = int(row["x0"])
        y0 = int(row["y0"])
        

        img = self.slide.read_region(location=(x0, y0),
                                        level=self.output_level,
                                        size=self.patch_size).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, patch_id,x0,y0
           


    def __del__(self):
        if hasattr(self, "slide"):
            self.slide.close()