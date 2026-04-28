# Swin Transformers for Slide-Level Leukaemia Classification with Attention Maps - Application

## Overview
This application performs slide-level leukaemia classification from whole slide bone marrow images using pretrained Swin Transformer models.

It replicates the two-stage pipeline from the training repository:
1. ROI detection (224×224 patches)
2. Slide-level subtype classification (512×512 patches)

The app is designed for inference and visualisation only.

**Disclaimer:**
This application is intended for research and demonstration purposes only and is not suitable for clinical use.

## Application Demo

A deployed version of the pipeline is available as an interactive Streamlit application:

[streamlit demo app](https://leukaemia-subtype-classification-demo-app.streamlit.app/)

The demo allows users to run inference on precomputed whole slide samples and visualise model predictions and attention maps.


## Relation to Training Repository
This application is built on top of the full training and data processing pipeline available here:

[link to training repository](https://github.com/IngunnOfs/Swin-Transformers-for-Slide-Level-Leukaemia-Classification-with-Attention-Maps)

The referenced repository contains:
- Data preprocessing and patch extraction
- Model training and evaluation
- Attention map generation methodology

This app uses:
- Pretrained model weights
- Preprocessed pipeline logic (adapted for inference)

Only the components required for inference and visualisation are included here.

## Dataset

This app expects preprocessed `.tif` whole slide images.


- Precomputed results from three slides (AML, ALL and  CML) are included for the demo application.
- Because of the data size no samples are provided for full application testing.
- The full dataset must be downloaded separately and converted to `.tif` pyramid files prior to use.


Original dataset:

Höfener, H., Kock, F., Pontones, M. A., Ghete, T., Pfrang, D., Dickel, N., Kunz, M., Schacherer,
D., Clunie, D. A., Fedorov, A., Westphal, M., & Metzler, M. (2025, September).
*BoneMarrowWSI-PediatricLeukemia: A Comprehensive Dataset of Bone Marrow Aspirate
Smear Whole Slide Images with Expert Annotations and Clinical Data in Pediatric
Leukemia.* https://doi.org/10.5281/ZENODO.14933087

The full dataset can be found at the National Cancer Institutes Imaging Data Commons [HERE](https://portal.imaging.datacommons.cancer.gov/explore/filters/?collection_id=Community&collection_id=bonemarrowwsi_pediatricleukemia)


Reference:

Fedorov, A., Longabaugh, W. J. R., Pot, D., Clunie, D. A., Pieper, S., Aerts, H. J. W. L., Homeyer, A., Lewis, R., Akbarzadeh, A., Bontempi, D., Clifford, W., Herrmann, M. D., Höfener, H., Octaviano, I., Osborne, C., Paquette, S., Petts, J., Punzo, D., Reyes, M., Schacherer, D. P., Tian, M., White, G., Ziegler, E., Shmulevich, I., Pihl, T., Wagner, U., Farahani, K. & Kikinis, R.
*NCI Imaging Data Commons.* Cancer Res. 81, 4188–4193 (2021).
http://dx.doi.org/10.1158/0008-5472.CAN-21-0950 


## Project Structure

```
application/
|
|-- outputs/
|    |--patch and slide metadata (CSV files for ROI + cancer with and without labels)
|    |--wsi_thumbnail.png (thumbnail photo of whole slide with marked positive ROIs used for the cancer classification)
|    |--attention_images/
|    |    |--*.png (attention maps of top 5 confident patches)
|    |
|    |--demo_outputs/
|    |    |-- output .csv and .png files for demo app (pre-computed)
|
|--models/
|    |--ROI_classification_model (PyTorch .pt)
|    |--CANCER_classification_model (PyTorch .pt)
|
|--environments/
|    |--*.yml (required environments)
|
|--pipeline/
|    |--*.py (pipeline functions used by the scripts)
|--scripts/
|    |--*.py (script files used by the app)
|
|--sample_wsi/
|    |--*.tif (must be added with downloaded converted files)
|
|--app.py (full app for classification of whole slide images)
|
|- app_DEMO.py (demo app runnable without storage, environment or computational constraints)

```

## Outputs

- Slide-level leukaemia subtype prediction
- Thumbnail whole slide image with highlighted (green) positive ROI patches which the model base its prediction on
- Top 5 most confident suptype predicted patches with cumulative attention map overlay
- Confident scores of predicted leukaemia subtype

## Model Weights
This repository includes pretrained models:

- ROI classification model (Swin-S)
- Cancer subtype classification model (Swin-B)

These models were trained using the pipeline described in the training repository.
Because of their size, they are available for download [HERE](https://doi.org/10.5281/zenodo.19854417)



## Running the app



**Overview over app stages**
1. First patching (224x224 at 20x resolution)
2. ROI labelling the patches using ROI_classification_model
3. Second patching (512x512 at full resolution)
4. SLide-level leukaemia classification
5. Generate top 5 most confident patches with attention map overlay


### Installation:
This project uses multiple conda environments because of dependency conflicts:
 - wsi_env (preprocessing and patching WSIs)
 - swin_gpu (model training and testing)
 - gradcam_env (attention maps)
 - streamlit_env (running app.py and app_DEMO.py)

```
Create environments:
 conda env create -f wsi_env.yml
 conda env create -f swin_env.yml
 conda env create -f gradacam_env.yml
 conda env create -f streamlit_env.yml
```

### Running the app:

Download whole slide images from references and convert to `.tif` format following the original pipeline. 
Save the downloaded images inside a \sample_wsi folder within the application folder to ensure the dropdown menu finds the images.

Download the models from the provided link above and save them in the application directory `application\models\`.

Activate the streamlit environment:

`conda activate streamlit_env`

Run:

`streamlit run app.py`

Then open the provided local URL in your browser.

Select a `.tif` whole slide image and click **Run pipeline**.


### Runtime Notes:

- Full pipeline: **~25 minutes per slide** (depending on hardware)
- Demo app: runs instantly using precomputed outputs






