import streamlit as st
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from huggingface_hub import hf_hub_download
import os
from PIL import Image
import io
import gc

os.environ["LANG"] = "en_US.UTF-8"
st.set_page_config(page_title="FaceVortex | AI Studio", page_icon="🌀", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3, label, .stMarkdown p { color: #ffffff !important; }
    div[data-testid="stFileUploader"] { background-color: #161b22; border: 1px dashed #30363d; border-radius: 10px; padding: 10px; }
    div[data-testid="stFileUploader"] section { background-color: #161b22 !important; }
    div[data-testid="stFileUploader"] section * { color: #c9d1d9 !important; }
    .stButton>button { width: 100%; background: linear-gradient(90deg, #ff4b4b 0%, #ff8c00 100%); color: white !important; font-weight: bold; border-radius: 8px; padding: 0.6rem; border: none; }
    .warning-badge { background-color: #1a1c23; border-left: 4px solid #ff4b4b; padding: 12px 15px; border-radius: 4px; margin-bottom: 20px; font-size: 14px; color: #ff9999 !important; border: 1px solid #2d3139; }
    .face-row { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🌀 FaceVortex Studio")
st.markdown("<div class='warning-badge'><b>WARNING (18+):</b> This application is strictly restricted to adult users for legal and authorized content only.</div>", unsafe_allow_html=True)

def resize_img(img, max_dim=800):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

@st.cache_resource
def load_models():
    model_file = hf_hub_download(repo_id="ezioruan/inswapper_128.onnx", filename="inswapper_128.onnx")
    app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(320, 320))
    swapper = insightface.model_zoo.get_model(model_file, providers=['CPUExecutionProvider'])
    return app, swapper

with st.spinner("⏳ Pokretanje AI motora (prvi put traje oko 30 sekundi)..."):
    try:
        face_app, swapper = load_models()
    except Exception as e:
        st.error(f"Greška: {e}")
        st.stop()

st.markdown("### Select Operating Mode")
app_mode = st.radio("Mode:", ["1. Single Face Swap", "2. Multi-Face Swap"], label_visibility="collapsed")
st.markdown("---")

if "Single" in app_mode:
    col1, col2 = st.columns(2)
    with col1:
        t_file = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"], key="t_s")
    with col2:
        s_file = st.file_uploader("Upload Source Face", type=["jpg", "jpeg", "png"], key="s_s")

    if t_file and s_file:
        img_t = resize_img(cv2.imdecode(np.frombuffer(t_file.getvalue(), np.uint8), cv2.IMREAD_COLOR), 800)
        img_s = resize_img(cv2.imdecode(np.frombuffer(s_file.getvalue(), np.uint8), cv2.IMREAD_COLOR), 600)

        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB), caption="Target", use_container_width=True)
        c2.image(cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB), caption="Source", use_container_width=True)

        if st.button("🚀 Run Single Face Swap"):
            with st.spinner("Processing..."):
                t_faces = face_app.get(img_t)
                s_faces = face_app.get(img_s)
                if not t_faces or not s_faces:
                    st.error("Lice nije pronađeno na jednoj od slika.")
                else:
                    res = img_t.copy()
                    for tf in t_faces:
                        res = swapper.get(res, tf, s_faces, paste_back=True)
                    final_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                    st.image(final_rgb, caption="Result", use_container_width=True)
                    buf = io.BytesIO()
                    Image.fromarray(final_rgb).save(buf, format="JPEG", quality=95)
                    st.download_button("📥 Download Result", buf.getvalue(), "result.jpg", "image/jpeg")
                    gc.collect()
else:
    col_l, col_r = st.columns(2)
    reps = {}
    img_m = None
    faces_m = []

    with col_l:
        t_file_m = st.file_uploader("Upload group image", type=["jpg", "jpeg", "png"], key="t_m")
        if t_file_m:
            img_m = resize_img(cv2.imdecode(np.frombuffer(t_file_m.getvalue(), np.uint8), cv2.IMREAD_COLOR), 800)
            faces_m = face_app.get(img_m)
            st.image(cv2.cvtColor(img_m, cv2.COLOR_BGR2RGB), caption=f"Detected {len(faces_m)} faces", use_container_width=True)

    with col_r:
        if img_m is not None and len(faces_m) > 0:
            for i, f in enumerate(faces_m):
                st.markdown("<div class='face-row'>", unsafe_allow_html=True)
                b = f['bbox'].astype(int)
                h, w, _ = img_m.shape
                crop = img_m[max(0, b-20):min(h, b+20), max(0, b-20):min(w, b+20)]
                sc1, sc2, sc3 = st.columns([1, 0.3, 1.5])
                if crop.size > 0:
                    sc1.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), width=70, caption=f"#{i+1}")
                sc2.markdown("<h3 style='text-align:center;'>➡️</h3>", unsafe_allow_html=True)
                rf = sc3.file_uploader(f"New #{i+1}", type=["jpg", "jpeg", "png"], key=f"r_{i}")
                if rf:
                    ir = resize_img(cv2.imdecode(np.frombuffer(rf.getvalue(), np.uint8), cv2.IMREAD_COLOR), 600)
                    sf = face_app.get(ir)
                    if sf:
                        reps[i] = sf
                    else:
                        st.error(f"Nema lica #{i+1}")
                st.markdown("</div>", unsafe_allow_html=True)

    if img_m is not None and reps:
        if st.button("🚀 Run Multi-Face Swap"):
            with st.spinner("Processing..."):
                res = img_m.copy()
                for idx, sf in reps.items():
                    res = swapper.get(res, faces_m[idx], sf, paste_back=True)
                final_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                st.image(final_rgb, caption="Group Result", use_container_width=True)
                buf = io.BytesIO()
                Image.fromarray(final_rgb).save(buf, format="JPEG", quality=95)
                st.download_button("📥 Download Result", buf.getvalue(), "group.jpg", "image/jpeg")
                gc.collect()
