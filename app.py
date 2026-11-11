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

st.set_page_config(
    page_title="FaceVortex | AI Studio",
    page_icon="🌀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #ffffff !important;
    }
    .stRadio div[role="radiogroup"] label p {
        color: #ffffff !important;
        font-weight: bold;
    }
    div[data-testid="stFileUploader"] {
        background-color: #161b22;
        border: 1px dashed #30363d;
        border-radius: 10px;
        padding: 10px;
    }
    div[data-testid="stFileUploader"] section {
        background-color: #161b22 !important;
    }
    div[data-testid="stFileUploader"] section * {
        color: #c9d1d9 !important;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ff4b4b 0%, #ff8c00 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem;
        border: none;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #ff3333 0%, #ff7700 100%);
        color: white;
    }
    .warning-badge {
        background-color: #1a1c23;
        border-left: 4px solid #ff4b4b;
        padding: 12px 15px;
        border-radius: 4px;
        margin-bottom: 20px;
        font-size: 14px;
        color: #ff9999;
        border: 1px solid #2d3139;
    }
    .face-row {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌀 FaceVortex Studio")
st.markdown("<div class='warning-badge'><b>WARNING (18+):</b> This application is strictly restricted to adult users for legal and authorized content only.</div>", unsafe_allow_html=True)

# Funkcija za optimizaciju veličine slike da ne prepuni RAM memoriju servera
def resize_image_if_needed(img, max_dim=1024):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

model_path = "inswapper_128.onnx"
if not os.path.exists(model_path):
    with st.spinner("Downloading AI face swap model (first time only, please wait)..."):
        url = "https://huggingface.co"
        urllib.request.urlretrieve(url, model_path)

@st.cache_resource
def load_models():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1, det_size=(320, 320))
    swapper = insightface.model_zoo.get_model(model_path, download=False, download_zip=False)
    return app, swapper

with st.spinner("Initializing AI engine..."):
    face_app, swapper = load_models()

st.markdown("### Select Operating Mode")
app_mode = st.radio(
    "Choose your swap mode:",
    ["1. Single Face Swap (1 Target + 1 Source)", "2. Multi-Face Swap (Group Photo + Precise Mapping)"],
    label_visibility="collapsed"
)

st.markdown("---")

if "Single Face Swap" in app_mode:
    st.markdown("### 👤 Single Face Swap Mode")
    st.write("Upload a target image and a single source face to swap.")

    col1, col2 = st.columns(2)
    with col1:
        target_file_s = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"], key="target_single")
    with col2:
        source_file_s = st.file_uploader("Upload Source Face", type=["jpg", "jpeg", "png"], key="source_single")

    if target_file_s is not None and source_file_s is not None:
        img_target = cv2.imdecode(np.frombuffer(target_file_s.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        img_source = cv2.imdecode(np.frombuffer(source_file_s.getvalue(), np.uint8), cv2.IMREAD_COLOR)

        img_target = resize_image_if_needed(img_target, max_dim=1024)
        img_source = resize_image_if_needed(img_source, max_dim=800)

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.image(cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB), caption="Target Image", use_container_width=True)
        with pcol2:
            st.image(cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB), caption="Source Face", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Run Single Face Swap"):
            with st.spinner("Processing single face swap..."):
                t_faces = face_app.get(img_target)
                s_faces = face_app.get(img_source)

                if len(t_faces) == 0:
                    st.error("No face detected in the target image.")
                elif len(s_faces) == 0:
                    st.error("No face detected in the source image.")
                else:
                    res_img = img_target.copy()
                    src_face = s_faces[0]
                    for t_face in t_faces:
                        res_img = swapper.get(res_img, t_face, src_face, paste_back=True)

                    st.markdown("### ✨ Result Output")
                    st.success("Face swap completed successfully!")
                    
                    final_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                    st.image(final_rgb, caption="Final Result", use_container_width=True)

                    pil_img = Image.fromarray(final_rgb)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG", quality=95)
                    byte_im = buf.getvalue()

                    st.download_button(
                        label="📥 Download Result Image",
                        data=byte_im,
                        file_name="facevortex_single.jpg",
                        mime="image/jpeg"
                    )
                    gc.collect()

else:
    st.markdown("### 👥 Multi-Face Swap Mode")
    st.write("Upload a group target image, then match each detected person precisely.")

    col_left, col_right = st.columns(2)
    face_replacements = {}
    image_target_m = None
    target_faces_m = []

    with col_left:
        st.markdown("#### Target Group Image")
        target_file_m = st.file_uploader("Upload target group image", type=["jpg", "jpeg", "png"], key="target_multi")
        
        if target_file_m is not None:
            image_target_m = cv2.imdecode(np.frombuffer(target_file_m.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            image_target_m = resize_image_if_needed(image_target_m, max_dim=1024)
            target_faces_m = face_app.get(image_target_m)
            st.image(cv2.cvtColor(image_target_m, cv2.COLOR_BGR2RGB), caption=f"Target (Detected {len(target_faces_m)} faces)", use_container_width=True)

    with col_right:
        st.markdown("#### Face List & Replacements")
        if target_file_m is not None and len(target_faces_m) > 0:
            st.write("Match each detected face with a new source face:")
            
            for i, face in enumerate(target_faces_m):
                st.markdown("<div class='face-row'>", unsafe_allow_html=True)
                
                bbox = face['bbox'].astype(int)
                h, w, _ = image_target_m.shape
                x1, y1 = max(0, bbox[0] - 20), max(0, bbox[1] - 20)
                x2, y2 = min(w, bbox[2] + 20), min(h, bbox[3] + 20)
                face_crop = image_target_m[y1:y2, x1:x2]
                
                subcol1, subcol2, subcol3 = st.columns([1, 0.3, 1.5])
                with subcol1:
                    if face_crop.size > 0:
                        st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), width=70, caption=f"Face #{i+1}")
                with subcol2:
                    st.markdown("<h3 style='text-align: center; margin-top: 15px;'>➡️</h3>", unsafe_allow_html=True)
                with subcol3:
                    rep_file = st.file_uploader(f"New face #{i+1}", type=["jpg", "jpeg", "png"], key=f"rep_{i}")
                    if rep_file is not None:
                        img_rep = cv2.imdecode(np.frombuffer(rep_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                        img_rep = resize_image_if_needed(img_rep, max_dim=800)
                        s_faces = face_app.get(img_rep)
                        if len(s_faces) > 0:
                            face_replacements[i] = s_faces[0]
                        else:
                            st.error(f"No face detected in upload #{i+1}")
                st.markdown("</div>", unsafe_allow_html=True)

    if image_target_m is not None and len(face_replacements) > 0:
        st.markdown("---")
        if st.button("🚀 Run Multi-Face Swap"):
            with st.spinner("Processing multi-face swap..."):
                res_img = image_target_m.copy()
                
                for idx, src_face in face_replacements.items():
                    t_face = target_faces_m[idx]
                    res_img = swapper.get(res_img, t_face, src_face, paste_back=True)
                
                st.markdown("### ✨ Multi-Face Result Output")
                st.success("All selected faces replaced successfully!")
                
                final_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                st.image(final_rgb, caption="Final Group Result", use_container_width=True)
                
                pil_img = Image.fromarray(final_rgb)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=95)
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Group Result",
                    data=byte_im,
