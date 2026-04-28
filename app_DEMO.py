import streamlit as st
import subprocess
import json
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "scripts"
PIPELINE_DIR = BASE_DIR / "pipeline"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
SAMPLE_WSI_DIR = BASE_DIR / "sample_wsi"
DEMO_DIR = OUTPUTS_DIR / "demo_outputs"

# SCRIPTS PATHS
ROI_patching_script = SCRIPTS_DIR / "ROI_patching_script.py"
ROI_class_script =  SCRIPTS_DIR / "ROI_classification_script.py"
CANCER_patching_script =  SCRIPTS_DIR / "CANCER_patching_script.py"
CANCER_class_script =  SCRIPTS_DIR / "CANCER_classification_script.py"
ATTENTION_script = SCRIPTS_DIR /  "attention_script.py"

class_map = {0 : "AML", 1: "ALL", 2: "CML", 3: "MPAL"}


        
# TITLE
st.title("Whole slide image Leukaemia subtype classification")

st.markdown("**-- DEMO VERSION --**")

# list of demo options

demo_list = ["ALL_wsi_demo", "AML_wsi_demo", "CML_wsi_demo"]


# Dropdown selector
selected_wsi = st.selectbox("Select a WSI file", demo_list)


# locate slide specific results
if "ALL" in str(selected_wsi).upper():
    SLIDE_RESULTS_DIR = DEMO_DIR / "ALL"
elif "AML" in str(selected_wsi).upper():
    SLIDE_RESULTS_DIR = DEMO_DIR / "AML"
elif "CML" in str(selected_wsi).upper():
    SLIDE_RESULTS_DIR = DEMO_DIR / "CML"



# CONTROL
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False
if "results" not in st.session_state:
    st.session_state.results = {}

progress = st.progress(0)
status = st.empty()

# BUTTON
if st.button ("Run pipeline"):


    st.session_state.pipeline_done = False
    st.session_state.results = {}

    # -----------------------------------------------------------------------------------
    # 1.    FIRST PATCHING (224,224)
    # -----------------------------------------------------------------------------------
    status.text("Step 1/5: ROI patching...")
    progress.progress(10)


    first_patches_csv = SLIDE_RESULTS_DIR / "1_patches_224_224_20x.csv"




    # Store results
    st.session_state.results["first_patches_csv"] = first_patches_csv


    # -----------------------------------------------------------------------------------
    # 2.    ROI LABELLING PATCHES
    # -----------------------------------------------------------------------------------
    status.text("Step 2/5: ROI classification...")
    progress.progress(30)



    first_labelled_patches_csv = SLIDE_RESULTS_DIR / "2_patches_with_roi_predictions.csv"
    

    
    # Store results
    st.session_state.results["first_labelled_patches_csv"] = first_labelled_patches_csv

    # -----------------------------------------------------------------------------------
    # 3.    SECOND PATCHING (512,512)
    # -----------------------------------------------------------------------------------
    status.text("Step 3/5: Positive ROI patching...")
    progress.progress(50)

    second_patches_csv = SLIDE_RESULTS_DIR / "3_patches_512_512_40x.csv"
    wsi_thumbnail = SLIDE_RESULTS_DIR / "wsi_thumbnail_with_selected_rois.png"


    
    # Store results
    st.session_state.results["second_patches_csv"] = second_patches_csv
    st.session_state.results["wsi_thumbnail"] = wsi_thumbnail

    
    # -----------------------------------------------------------------------------------
    # 4.    LEUKAEMIA CLASSIFICATIOn, SLIDE LEVEL
    # -----------------------------------------------------------------------------------
    status.text("Step 4/5: Leukaemia classification of the whole slide...")
    progress.progress(70)

    second_patches_results_csv = SLIDE_RESULTS_DIR / "4_patches_with_cancer_predictions.csv"
    slide_results_csv = SLIDE_RESULTS_DIR / "4_slide_cancer_prediction.csv"


    
    # Store results
    st.session_state.results["second_patches_results_csv"] = second_patches_results_csv
    st.session_state.results["slide_results_csv"] = slide_results_csv

    # -----------------------------------------------------------------------------------
    # 5.    ATTENTION PLOTS
    # -----------------------------------------------------------------------------------
    status.text("Step 5/5: Generating attention maps...")
    progress.progress(90)

    figs_csv = SLIDE_RESULTS_DIR / "5_attention_results.csv"
  
    
    # Store results
    st.session_state.results["figs_csv"] = figs_csv
    progress.progress(100)
    status.text("Pipeline completed!")
    st.session_state.pipeline_done = True
    # -----------------------------------------------------------------------------------
    #  DISPLAY SECTION
    # -----------------------------------------------------------------------------------

    if st.session_state.pipeline_done:
        st.markdown("## <u>Slide Overview</u>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.markdown("### WSI Preview")
            st.markdown("**Selected ROIs in green overlay**")
            thumbnail_path = OUTPUTS_DIR / st.session_state.results["wsi_thumbnail"]
            st.image(str(thumbnail_path), use_container_width=True)

        with col2:
            st.markdown("### Prediction Results")

            # load slide-level results
            slide_df = pd.read_csv(st.session_state.results["slide_results_csv"])
            row = slide_df.iloc[0]

            prediction = row["prediction"]
            diagnosis = class_map[prediction]
            confidence = row["confidence"]
            confidence_perc = float(confidence) * 100

            st.markdown(f"**Diagnosis:** {diagnosis}")
            st.markdown(f"**Confidence:** {confidence_perc:.1f}%")

            st.progress(float(confidence))

        st.markdown("---")
        st.subheader("Top 5 Most Confident Attention Maps")

        figs_df = pd.read_csv(st.session_state.results["figs_csv"])

        for _, row in figs_df.iterrows():
            relative_path = str(row["fig_path"])
            full_path = OUTPUTS_DIR / relative_path
            st.image(full_path, use_container_width=True)


    



        
    
