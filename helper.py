import numpy as np
import pandas as pd
import json
import os
import difflib

def compute_user_vector(indices, E_img):
    vecs = E_img[indices]          
    u = vecs.mean(axis=0)
    u = u / np.linalg.norm(u)
    return u

def rank_characters(u, char_embeds):
    scores = [(cid, np.matmul(u,v)) for cid, v in char_embeds.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def rank_labels(u, label_embeds, baseline):
    scores = []
    for lab, vec in label_embeds.items():
        sim = float(u @ vec) - baseline[lab]
        scores.append((lab, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def find_closest_image(name, folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]

    if not files:
        return None
    best_match = difflib.get_close_matches(name.lower(), [f.lower() for f in files], n=1, cutoff=0.0)

    if best_match:
        for f in files:
            if f.lower() == best_match[0]:
                return os.path.join(folder, f)

    return None