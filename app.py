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


# SCRIPTS PATHS
ROI_patching_script = SCRIPTS_DIR / "ROI_patching_script.py"
ROI_class_script =  SCRIPTS_DIR / "ROI_classification_script.py"
CANCER_patching_script =  SCRIPTS_DIR / "CANCER_patching_script.py"
CANCER_class_script =  SCRIPTS_DIR / "CANCER_classification_script.py"
ATTENTION_script = SCRIPTS_DIR /  "attention_script.py"

class_map = {0 : "AML", 1: "ALL", 2: "CML", 3: "MPAL"}

# ENVIRONMENT PATHS 
wsi_env = "wsi-env"
swin_gpu = "swin_gpu"
gradcam_env = "gradcam_env"

def run_in_env(env_name, script, args):
    return subprocess.run(
        ["conda", "run", "-n", env_name, "python", script] + args,
        capture_output=True,
        text=True
    )

# TITLE
st.title("Whole Slide Image Leukaemia Subtype Classification")
st.markdown("*Note: processing time can take approximately 25 minutes...*")

# Find all tif files recursively
wsi_files = list(SAMPLE_WSI_DIR.rglob("*.tif"))

# Convert to strings
wsi_map = {p.name: p for p in wsi_files}

# Dropdown selector
selected_wsi = st.selectbox("Select a WSI file", list(wsi_map.keys()))

# Clean path automatically
wsi_path = wsi_map[selected_wsi]
wsi_path = Path(wsi_path).resolve()



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

    result1 = run_in_env(wsi_env,str(ROI_patching_script), [str(wsi_path)])
    
    progress.progress(20)

    if result1.returncode != 0:
        st.error(result1.stderr)
    else:
        stdout1 = result1.stdout.strip()
        outputs1 = json.loads(stdout1)
        first_patches_csv = outputs1["patches_csv"]
        
        status.text("Step 1/5: ROI patching completed.")

    # Store results
    st.session_state.results["first_patches_csv"] = first_patches_csv


    # -----------------------------------------------------------------------------------
    # 2.    ROI LABELLING PATCHES
    # -----------------------------------------------------------------------------------
    status.text("Step 2/5: ROI classification...")
    progress.progress(30)

    result2 = run_in_env(swin_gpu, str(ROI_class_script), [str(wsi_path),first_patches_csv])

    progress.progress(40)

    if result2.returncode != 0:
        st.error(result2.stderr)
    else:
        stdout2 = result2.stdout.strip()
        outputs2 = json.loads(stdout2)
        first_labelled_patches_csv = outputs2["patches_csv"]

        status.text("Step 2/5: ROI classification completed.")
    
    # Store results
    st.session_state.results["first_labelled_patches_csv"] = first_labelled_patches_csv

    # -----------------------------------------------------------------------------------
    # 3.    SECOND PATCHING (512,512)
    # -----------------------------------------------------------------------------------
    status.text("Step 3/5: Positive ROI patching...")
    progress.progress(50)

    result3 = run_in_env(wsi_env, str(CANCER_patching_script), [str(wsi_path),first_labelled_patches_csv])

    progress.progress(60)

    if result3.returncode != 0:
        st.error(result3.stderr)
    else:
        stdout3 = result3.stdout.strip()
        outputs3 = json.loads(stdout3)
        second_patches_csv = (outputs3["patches_csv"])
        wsi_thumbnail = outputs3['wsi_thumbnail']
        
        status.text("Step 3/5: Positive ROI patching completed.")
    
    # Store results
    st.session_state.results["second_patches_csv"] = second_patches_csv
    st.session_state.results["wsi_thumbnail"] = wsi_thumbnail

    
    # -----------------------------------------------------------------------------------
    # 4.    LEUKAEMIA CLASSIFICATIOn, SLIDE LEVEL
    # -----------------------------------------------------------------------------------
    status.text("Step 4/5: Leukaemia classification of the whole slide...")
    progress.progress(70)

    result4 = run_in_env(swin_gpu, str(CANCER_class_script),[str(wsi_path),second_patches_csv])
    

    progress.progress(80)
    if result4.returncode != 0:
        st.error(result4.stderr)
    else:
        stdout = result4.stdout.strip()
        outputs = json.loads(stdout)
        second_patches_results_csv = outputs["patch_csv"]
        slide_results_csv = outputs["slide_csv"]
        status.text("Step 4/5: Leukaemia classification completed.")
    
    # Store results
    st.session_state.results["second_patches_results_csv"] = second_patches_results_csv
    st.session_state.results["slide_results_csv"] = slide_results_csv

    

    # -----------------------------------------------------------------------------------
    # 5.    ATTENTION PLOTS
    # -----------------------------------------------------------------------------------
    status.text("Step 5/5: Generating attention maps...")
    progress.progress(90)

    result5 = run_in_env(gradcam_env, str(ATTENTION_script),[str(wsi_path), second_patches_results_csv, slide_results_csv])

    progress.progress(100)
    if result5.returncode != 0:
        st.error(result5.stderr)
    else:
        stdout5 = result5.stdout.strip()
        outputs5 = json.loads(stdout5)
        figs_csv = outputs5["figs_csv"]

        status.text("Step 5/5: attention maps generated.")
    
    # Store results
    st.session_state.results["figs_csv"] = figs_csv
    progress.progress(100)
    status.text("Pipeline completed!")
    st.session_state.pipeline_done = True

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
        st.markdown("### Top 5 Most Confident Attention Maps")

        figs_df = pd.read_csv(st.session_state.results["figs_csv"])

        for _, row in figs_df.iterrows():
            relative_path = str(row["fig_path"])
            full_path = OUTPUTS_DIR / relative_path
            st.image(full_path, use_container_width=True)

        

        