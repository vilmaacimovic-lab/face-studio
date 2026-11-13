import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from huggingface_hub import hf_hub_download
import os, io, gc
from PIL import Image, ImageSequence

os.environ["LANG"] = "en_US.UTF-8"
st.set_page_config(page_title="FaceVortex | AI Studio", page_icon="🌀", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""<style>
.stApp { background-color: #0e1117; }
header[data-testid="stHeader"], #MainMenu, footer { visibility: hidden; display: none; }
h1, h2, h3, label, .stMarkdown p { color: #ffffff !important; }
div[data-testid="stFileUploader"] { background-color: #161b22; border: 1px dashed #30363d; border-radius: 10px; padding: 10px; }
div[data-testid="stFileUploader"] section { background-color: #161b22 !important; }
div[data-testid="stFileUploader"] section * { color: #c9d1d9 !important; }
.stButton>button { width: 100%; background: linear-gradient(90deg, #ff4b4b 0%, #ff8c00 100%); color: white !important; font-weight: bold; border-radius: 8px; padding: 0.6rem; border: none; }
.warning-badge { background-color: #1a1c23; border-left: 4px solid #ff4b4b; padding: 12px 15px; border-radius: 4px; margin-bottom: 15px; font-size: 14px; color: #ff9999 !important; border: 1px solid #2d3139; }
.face-row { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; margin-bottom: 10px; }
.custom-footer { text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #30363d; color: #8b949e !important; font-size: 13px; }
</style>""", unsafe_allow_html=True)

def show_ad():
    components.html("""<div style="display:flex;justify-content:center;width:100%;overflow:hidden;">
    <script>atOptions={'key':'61148e63c979b55d9e52794e62fd2799','format':'iframe','height':90,'width':728,'params':{}};</script>
    <script src="https://biomanos.org"></script></div>""", height=105)

st.title("🌀 FaceVortex Studio")
st.markdown("<div class='warning-badge'><b>WARNING (18+):</b> This application is strictly restricted to adult users for legal and authorized content only.</div>", unsafe_allow_html=True)
show_ad()

def resize_img(img, max_dim=800):
    h, w, _ = img.shape
    if max(h, w) > max_dim:
        s = max_dim / float(max(h, w))
        return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img

@st.cache_resource
def load_models():
    mf = hf_hub_download(repo_id="ezioruan/inswapper_128.onnx", filename="inswapper_128.onnx")
    rf = hf_hub_download(repo_id="public-data/insightface", filename="models/buffalo_l/w600k_r50.onnx")
    app = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection'], providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(320, 320))
    rm = insightface.model_zoo.get_model(rf, providers=['CPUExecutionProvider'])
    rm.prepare(ctx_id=-1)
    app.models['recognition'] = rm
    sw = insightface.model_zoo.get_model(mf, providers=['CPUExecutionProvider'])
    gc.collect()
    return app, sw

with st.spinner("⏳ Učitavanje AI motora..."):
    face_app, swapper = load_models()

st.markdown("### Select Operating Mode")
app_mode = st.radio("Mode:", ["1. Single Face Swap", "2. Multi-Face Swap", "3. GIF Face Swap"], label_visibility="collapsed")
st.markdown("---")

if "Single" in app_mode:
    c1, c2 = st.columns(2)
    tf_in = c1.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"], key="t_s")
    sf_in = c2.file_uploader("Upload Source Face", type=["jpg", "jpeg", "png"], key="s_s")
    if tf_in and sf_in:
        img_t = resize_img(cv2.imdecode(np.frombuffer(tf_in.getvalue(), np.uint8), cv2.IMREAD_COLOR), 800)
        img_s = resize_img(cv2.imdecode(np.frombuffer(sf_in.getvalue(), np.uint8), cv2.IMREAD_COLOR), 600)
        p1, p2 = st.columns(2)
        p1.image(cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB), caption="Target", use_container_width=True)
        p2.image(cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB), caption="Source", use_container_width=True)
        if st.button("🚀 Run Single Face Swap"):
            with st.spinner("Processing..."):
                tfs, sfs = face_app.get(img_t), face_app.get(img_s)
                if not tfs or not sfs:
                    st.error("Lice nije pronađeno na jednoj od slika.")
                else:
                    src = next(iter(sfs))
                    res = img_t.copy()
                    for tf in tfs:
                        res = swapper.get(res, tf, src, paste_back=True)
                    rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                    st.image(rgb, caption="Result", use_container_width=True)
                    buf = io.BytesIO()
                    Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
                    st.download_button("📥 Download Result", buf.getvalue(), "result.jpg", "image/jpeg")
                    gc.collect()

elif "Multi" in app_mode:
    cl, cr = st.columns(2)
    reps, img_m, faces_m = {}, None, []
    with cl:
        tfm = st.file_uploader("Upload group image", type=["jpg", "jpeg", "png"], key="t_m")
        if tfm:
            img_m = resize_img(cv2.imdecode(np.frombuffer(tfm.getvalue(), np.uint8), cv2.IMREAD_COLOR), 800)
            faces_m = face_app.get(img_m)
            st.image(cv2.cvtColor(img_m, cv2.COLOR_BGR2RGB), caption=f"Detected {len(faces_m)} faces", use_container_width=True)
    with cr:
        if img_m is not None and len(faces_m) > 0:
            for i, f in enumerate(faces_m):
                st.markdown("<div class='face-row'>", unsafe_allow_html=True)
                x1, y1, x2, y2 = f['bbox'].astype(int).tolist()
                h, w, _ = img_m.shape
                crop = img_m[max(0, y1-20):min(h, y2+20), max(0, x1-20):min(w, x2+20)]
                s1, s2, s3 = st.columns([1, 0.3, 1.5])
                if crop.size > 0:
                    s1.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), width=70, caption=f"#{i+1}")
                s2.markdown("<h3 style='text-align:center;'>➡️</h3>", unsafe_allow_html=True)
                rf = s3.file_uploader(f"New #{i+1}", type=["jpg", "jpeg", "png"], key=f"r_{i}")
                if rf:
                    ir = resize_img(cv2.imdecode(np.frombuffer(rf.getvalue(), np.uint8), cv2.IMREAD_COLOR), 600)
                    sf = face_app.get(ir)
                    if sf: reps[i] = next(iter(sf))
                    else: st.error(f"Nema lica #{i+1}")
                st.markdown("</div>", unsafe_allow_html=True)
    if img_m is not None and reps:
        if st.button("🚀 Run Multi-Face Swap"):
            with st.spinner("Processing..."):
                res = img_m.copy()
                for idx, sf in reps.items():
                    res = swapper.get(res, faces_m[idx], sf, paste_back=True)
                rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                st.image(rgb, caption="Group Result", use_container_width=True)
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
                st.download_button("📥 Download Result", buf.getvalue(), "group.jpg", "image/jpeg")
                gc.collect()

else:
    st.markdown("### 🎞️ GIF Face Swap Mode (Turbo)")
    c1, c2 = st.columns(2)
    gf = c1.file_uploader("Upload Target GIF", type=["gif"], key="t_gif")
    sgf = c2.file_uploader("Upload Source Face", type=["jpg", "jpeg", "png"], key="s_gif")
    if gf and sgf:
        p1, p2 = st.columns(2)
        p1.image(gf, caption="Target GIF", use_container_width=True)
        isg = resize_img(cv2.imdecode(np.frombuffer(sgf.getvalue(), np.uint8), cv2.IMREAD_COLOR), 600)
        p2.image(cv2.cvtColor(isg, cv2.COLOR_BGR2RGB), caption="Source Face", use_container_width=True)
        if st.button("🚀 Run GIF Face Swap"):
            sfs = face_app.get(isg)
            if not sfs:
                st.error("Lice nije pronađeno na izvornoj slici.")
            else:
                src = next(iter(sfs))
                pg = Image.open(gf)
                frames, durs = [], []
                all_f = [f.copy() for f in ImageSequence.Iterator(pg)]
                step = 2 if len(all_f) > 15 else 1
                sel = all_f[::step][:45]
                pb, stxt = st.progress(0), st.empty()
                for idx, fr in enumerate(sel):
                    stxt.text(f"Obrada sličice {idx + 1} od {len(sel)}...")
                    durs.append(fr.info.get('duration', 100) * step)
                    bf = resize_img(cv2.cvtColor(np.array(fr.convert('RGB')), cv2.COLOR_RGB2BGR), 450)
                    for tf in face_app.get(bf):
                        bf = swapper.get(bf, tf, src, paste_back=True)
                    frames.append(Image.fromarray(cv2.cvtColor(bf, cv2.COLOR_BGR2RGB)))
                    pb.progress((idx + 1) / len(sel))
                stxt.text("Spajanje gotovog GIF-a...")
                gbuf = io.BytesIO()
                next(iter(frames)).save(gbuf, format="GIF", save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=False)
                stxt.empty()
                st.success("GIF zamena lica uspešno završena!")
                st.image(gbuf.getvalue(), caption="Final GIF Result", use_container_width=True)
                st.download_button("📥 Download Result GIF", gbuf.getvalue(), "facevortex_result.gif", "image/gif")
                gc.collect()

st.markdown("<br>", unsafe_allow_html=True)
show_ad()
st.markdown("<div class='custom-footer'>© 2026 FaceVortex Studio | SEXYMAN. All rights reserved.</div>", unsafe_allow_html=True)
