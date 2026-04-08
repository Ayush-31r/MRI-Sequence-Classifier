import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
import tempfile
import os
from preprocess import preprocess

MODEL_PATH = "models/model.pth"
CLASSES    = ["bold", "dwi", "flair", "t1w", "t2w"]
THRESHOLD  = 0.75
DEVICE     = torch.device("cpu")

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict(filepath, model):
    tensor = preprocess(filepath).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().numpy()

    pred_idx   = probs.argmax()
    probability = float(probs[pred_idx])
    label      = CLASSES[pred_idx]
    uncertain  = probability < THRESHOLD
    prob_dict  = {cls: float(p) for cls, p in zip(CLASSES, probs)}

    return label, probability, uncertain, prob_dict

st.set_page_config(page_title="Brain MRI Classifier")
st.title("Brain MRI Sequence Classifier")
st.caption("Classifies NIfTI scans into: T1w, T2w, FLAIR, DWI, BOLD")

model = load_model()

uploaded = st.file_uploader("Upload a NIfTI file", type=["nii", "gz"])

if uploaded:
    suffix = ".nii.gz" if uploaded.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Running inference..."):
        try:
            label, probability, uncertain, prob_dict = predict(tmp_path, model)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()
        finally:
            os.unlink(tmp_path)

    if uncertain:
        st.warning(f"Uncertain — probability below {THRESHOLD:.0%}")
    else:
        st.success(f"Predicted: {label.upper()}")

    col1, col2 = st.columns(2)
    col1.metric("Label", label.upper())
    col2.metric("Probability", f"{probability:.1%}")

    st.subheader("Class Probabilities")
    st.bar_chart(prob_dict)