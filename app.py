import streamlit as st
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import urllib.request
from PIL import Image
import io
import gc

os.environ["LANG"] = "en_US.UTF-8"
st.set_page_config(page_title="FaceVortex | AI Studio", page_icon="🌀", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    h1, h2, h3, h4, h5, h6, p, span, label, div { color: #ffffff !important; }
    .stRadio div[role="radiogroup"] label p { color: #ffffff !important; font-weight: bold; }
    div[data-testid="stFileUploader"] { background-color: #161b22; border: 1px dashed #30363d; border-radius: 10px; padding: 10px; }
    div[data-testid="stFileUploader"] section { background-color: #161b22 !important; }
    div[data-testid="stFileUploader"] section * { color: #c9d1d9 !important; }
    .stButton>button { width: 100%; background: linear-gradient(90deg, #ff4b4b 0%, #ff8c00 100%); color: white; font-weight: bold; border-radius: 8px; padding: 0.6rem; border: none; }
    .warning-badge { background-color: #1a1c23; border-left: 4px solid #ff4b4b; padding: 12px 15px; border-radius: 4px; margin-bottom: 20px; font-size: 14px; color: #ff9999; border: 1px solid #2d3139; }
    .face-row { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🌀 FaceVortex Studio")
st.markdown("<div class='warning-badge'><b>WARNING (18+):</b> This application is strictly restricted to adult users for legal and authorized content only.</div>", unsafe_allow_html=True)

def resize_image_if_needed(img, max_dim=800):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

model_path = "inswapper_128.onnx"
if not os.path.exists(model_path):
    st.info("⏳ Preuzimanje AI modela (oko 500MB, samo prvi put, sačekajte oko 30 sekundi)...")
    url = "https://huggingface.co"
    urllib.request.urlretrieve(url, model_path)
    st.rerun()

@st.cache_resource
def load_models():
    app = FaceAnalysis(name='buffalo_sc')
    app.prepare(ctx_id=-1, det_size=(320, 320))
    swapper = insightface.model_zoo.get_model(model_path, download=False, download_zip=False)
    return app, swapper

face_app, swapper = load_models()

st.markdown("### Select Operating Mode")
app_mode = st.radio("Mode:", ["1. Single Face Swap (1 Target + 1 Source)", "2. Multi-Face Swap (Group Photo + Precise Mapping)"], label_visibility="collapsed")
st.markdown("---")

if "Single Face Swap" in app_mode:
    st.markdown("### 👤 Single Face Swap Mode")
    col1, col2 = st.columns(2)
    with col1:
        target_file_s = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"], key="target_single")
    with col2:
        source_file_s = st.file_uploader("Upload Source Face", type=["jpg", "jpeg", "png"], key="source_single")

    if target_file_s is not None and source_file_s is not None:
        img_target = resize_image_if_needed(cv2.imdecode(np.frombuffer(target_file_s.getvalue(), np.uint8), cv2.IMREAD_COLOR), 800)
        img_source = resize_image_if_needed(cv2.imdecode(np.frombuffer(source_file_s.getvalue(), np.uint8), cv2.IMREAD_COLOR), 600)

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.image(cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB), caption="Target Image", use_container_width=True)
        with pcol2:
            st.image(cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB), caption="Source Face", use_container_width=True)

        if st.button("🚀 Run Single Face Swap"):
            with st.spinner("Processing..."):
                t_faces = face_app.get(img_target)
                s_faces = face_app.get(img_source)
                if len(t_faces) == 0 or len(s_faces) == 0:
                    st.error("No face detected in one of the images.")
                else:
                    res_img = img_target.copy()
                    for t_face in t_faces:
                        res_img = swapper.get(res_img, t_face, s_faces[0], paste_back=True)
                    st.success("Face swap completed!")
                    final_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                    st.image(final_rgb, caption="Final Result", use_container_width=True)
                    buf = io.BytesIO()
                    Image.fromarray(final_rgb).save(buf, format="JPEG", quality=95)
                    st.download_button(label="📥 Download Result", data=buf.getvalue(), file_name="facevortex_single.jpg", mime="image/jpeg")
                    gc.collect()
else:
    st.markdown("### 👥 Multi-Face Swap Mode")
    col_left, col_right = st.columns(2)
    face_replacements = {}
    image_target_m = None
    target_faces_m = []

    with col_left:
        target_file_m = st.file_uploader("Upload target group image", type=["jpg", "jpeg", "png"], key="target_multi")
        if target_file_m is not None:
            image_target_m = resize_image_if_needed(cv2.imdecode(np.frombuffer(target_file_m.getvalue(), np.uint8), cv2.IMREAD_COLOR), 800)
            target_faces_m = face_app.get(image_target_m)
            st.image(cv2.cvtColor(image_target_m, cv2.COLOR_BGR2RGB), caption=f"Detected {len(target_faces_m)} faces", use_container_width=True)

    with col_right:
        if target_file_m is not None and len(target_faces_m) > 0:
            for i, face in enumerate(target_faces_m):
                st.markdown("<div class='face-row'>", unsafe_allow_html=True)
                bbox = face['bbox'].astype(int)
                h, w, _ = image_target_m.shape
                face_crop = image_target_m[max(0, bbox[1]-20):min(h, bbox[3]+20), max(0, bbox[0]-20):min(w, bbox[2]+20)]
                subcol1, subcol2, subcol3 = st.columns([1, 0.3, 1.5])
                with subcol1:
                    if face_crop.size > 0:
                        st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), width=70, caption=f"Face #{i+1}")
                with subcol2:
                    st.markdown("<h3 style='text-align: center; margin-top: 15px;'>➡️</h3>", unsafe_allow_html=True)
                with subcol3:
                    rep_file = st.file_uploader(f"New face #{i+1}", type=["jpg", "jpeg", "png"], key=f"rep_{i}")
                    if rep_file is not None:
                        img_rep = resize_image_if_needed(cv2.imdecode(np.frombuffer(rep_file.getvalue(), np.uint8), cv2.IMREAD_COLOR), 600)
                        s_faces = face_app.get(img_rep)
                        if len(s_faces) > 0:
                            face_replacements[i] = s_faces[0]
                        else:
                            st.error(f"No face in #{i+1}")
                st.markdown("</div>", unsafe_allow_html=True)

    if image_target_m is not None and len(face_replacements) > 0:
        if st.button("🚀 Run Multi-Face Swap"):
            with st.spinner("Processing..."):
                res_img = image_target_m.copy()
                for idx, src_face in face_replacements.items():
                    res_img = swapper.get(res_img, target_faces_m[idx], src_face, paste_back=True)
                st.success("All selected faces replaced!")
                final_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                st.image(final_rgb, caption="Final Group Result", use_container_width=True)
                buf = io.BytesIO()
                Image.fromarray(final_rgb).save(buf, format="JPEG", quality=95)
                st.download_button(label="📥 Download Group Result", data=buf.getvalue(), file_name="facevortex_multi.jpg", mime="image/jpeg")
                gc.collect()
