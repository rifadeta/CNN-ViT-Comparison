import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Lettuce Disease Classifier", layout="centered")

st.title("🥬 Klasifikasi Penyakit Daun Selada")
st.write("Unggah foto daun selada untuk mendeteksi apakah sehat atau terkena penyakit.")

# --- LOAD MODEL (DI-CACHE AGAR CEPAT) ---
@st.cache_resource
def load_prediction_model(model_name):
    path = "cnn_model" if model_name == "CNN Model" else "vit_model"
    model = tf.keras.Sequential([
        tf.keras.layers.TFSMLayer(path, call_endpoint="serve")
    ])
    return model

# --- SIDEBAR: PILIH MODEL ---
st.sidebar.header("Pengaturan Model")
selected_model_name = st.sidebar.selectbox(
    "Pilih Arsitektur Model:",
    ("CNN Model", "ViT Model")
)

model = load_prediction_model(selected_model_name)

# --- DAFTAR KELAS ---
class_names = ['Bacterial', 'Fungal', 'Healthy']

# --- UI UPLOAD GAMBAR ---
uploaded_file = st.file_uploader("Pilih gambar daun...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_container_width=True)

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

    # --- TAMPILKAN HASIL ---
    st.subheader(f"Hasil Prediksi: **{class_names[result_index]}**")
    st.progress(int(confidence))
    st.write(f"Tingkat Keyakinan: **{confidence:.2f}%**")

    # --- CATATAN TAMBAHAN (INI YANG BARU) ---
    st.info(
    "ℹ️ **Catatan:** Hasil prediksi berlaku untuk citra daun selada. "
    "Penggunaan pada objek lain dapat menghasilkan prediksi yang tidak akurat."
)