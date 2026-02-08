import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Lettuce Disease Classifier", layout="centered")

st.title("🥬 Klasifikasi Penyakit Daun Selada")
st.write("Unggah foto daun selada untuk mendeteksi apakah sehat atau terkena penyakit.")

CONFIDENCE_THRESHOLD = 70.0  # ambang batas keyakinan

# --- LOAD MODEL ---
@st.cache_resource
def load_prediction_model(model_name):
    path = "cnn_model" if model_name == "CNN Model" else "vit_model"
    model = tf.keras.Sequential([
        tf.keras.layers.TFSMLayer(path, call_endpoint="serve")
    ])
    return model

# --- SIDEBAR ---
st.sidebar.header("Pengaturan Model")
selected_model_name = st.sidebar.selectbox(
    "Pilih Arsitektur Model:",
    ("CNN Model", "ViT Model")
)

model = load_prediction_model(selected_model_name)

# --- KELAS ---
class_names = ['Bacterial', 'Fungal', 'Healthy']

# --- UPLOAD GAMBAR ---
uploaded_file = st.file_uploader(
    "Pilih gambar daun...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    st.write("---")
    st.write(f"**Prediksi menggunakan: {selected_model_name}**")

    # --- PREPROCESSING ---
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --- PREDIKSI ---
    preds = model.predict(img_array)

    if isinstance(preds, dict):
        prediction = list(preds.values())[0][0]
    else:
        prediction = preds[0]

    result_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # --- LOGIKA THRESHOLD ---
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Gambar tidak dapat diidentifikasi sebagai daun selada.")
        st.write(f"Tingkat keyakinan tertinggi: **{confidence:.2f}%**")
    else:
        st.success(f"Model digunakan: {selected_model_name}")
        st.subheader(f"Hasil Prediksi: **{class_names[result_index]}**")
        st.progress(int(confidence))
        st.write(f"Tingkat Keyakinan: **{confidence:.2f}%**")

    st.caption(
        "Catatan: Sistem hanya dirancang untuk mengklasifikasikan citra daun selada "
        "ke dalam tiga kelas (bacterial, fungal, healthy)."
    )
