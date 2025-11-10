import streamlit as st
import pandas as pd
import numpy as np
import json
from helper import *


st.set_page_config(
    page_title="Weak Hero Character Picker",
    layout="wide"
)

SAMPLE_SIZE = 20 
MIN_PICKS = 5

imgs_df   = pd.read_csv("/Users/alakarthika/Documents/Personal_Projects/WeakHero/assets/files/images.csv")
chars_df  = pd.read_csv("/Users/alakarthika/Documents/Personal_Projects/WeakHero/characters.csv")
labels_df = pd.read_csv("/Users/alakarthika/Documents/Personal_Projects/WeakHero/explain_labels.csv")

E_img = np.load("/Users/alakarthika/Documents/Personal_Projects/WeakHero/assets/files/embeddings/images.npy")  
with open("/Users/alakarthika/Documents/Personal_Projects/WeakHero/assets/files/embeddings/char_embeds.json") as f:
    char_embeds = {k: np.array(v) for k, v in json.load(f).items()}

with open("/Users/alakarthika/Documents/Personal_Projects/WeakHero/assets/files/embeddings/label_embeds.json") as f:
    label_embeds = {k: np.array(v) for k, v in json.load(f).items()}

N_IMAGES = len(imgs_df)
u_global = E_img.mean(axis=0)
u_global /= np.linalg.norm(u_global)

baseline = {
    lab: float(u_global @ vec)
    for lab, vec in label_embeds.items()
}

IMAGES_PER_LABEL = 2

label_to_indices = {}
for label in labels_df["label"]:
    idxs = imgs_df.index[imgs_df["bucket"] == label].tolist()
    if idxs:
        label_to_indices[label] = idxs

def stratified_sample_indices():
    all_indices = []
    rng = np.random.default_rng()  

    for label in labels_df["label"]:
        idxs = label_to_indices.get(label, [])
        if len(idxs) == 0:
            continue
        if len(idxs) <= IMAGES_PER_LABEL:
            chosen = idxs  
        else:
            chosen = rng.choice(idxs, size=IMAGES_PER_LABEL, replace=False).tolist()
        all_indices.extend(chosen)

    return all_indices

if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = stratified_sample_indices()

if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = []

st.title("Which Weak Hero character are you?")
st.write("## Choose the pictures that catch your eye (minimum of 5).")
st.write("#### Support the Weak Hero Webtoon here! [Weak Hero](https://www.webtoons.com/en/action/weakhero/list?title_no=1726)")

if not st.session_state.submitted:
    selected = []
    cols = st.columns(5) 

    for i, idx in enumerate(st.session_state.sample_idx):
        row = imgs_df.loc[idx]
        col = cols[i % 5]
        with col:
            st.image(row["path"], use_container_width=True)
            checked = st.checkbox("Select", key=f"chk_{idx}")
            if checked:
                selected.append(idx)

    if st.button("Submit"):
        if len(selected) < MIN_PICKS:
            st.error(f"Please select at least {MIN_PICKS} images.")
        else:
            st.session_state.selected_idx = selected
            st.session_state.submitted = True
            st.rerun()

else:

    selected = st.session_state.selected_idx
    u = compute_user_vector(selected, E_img)

    # characters
    char_scores = rank_characters(u, char_embeds)
    top_char_id, top_score = char_scores[0]
    top_char_row = chars_df[chars_df["id"] == top_char_id].iloc[0]

    # labels (vibes)
    label_scores = rank_labels(u, label_embeds, baseline)
    top_vibes = label_scores[:3]

    st.header(f"Your Match: {top_char_row['name']}")
    st.caption(top_char_row["one_liner"])

    st.subheader("Your match profile~")
    for lab, sim in top_vibes:
        blurb = labels_df.loc[labels_df["label"] == lab, "blurb"].item()
        st.write(f"- **{lab}**: {blurb}")

    st.subheader("Your choices")
    cols = st.columns(5)
    for i, idx in enumerate(selected):
        with cols[i % 5]:
            st.image(imgs_df.loc[idx, "path"], use_container_width=True)

    if st.button("Retake"):
        # reset 
        for key in ["sample_idx", "submitted", "selected_idx"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
st.caption("This match is for fun and based on visual similarity using CLIP. No deep psychology here :)")