import streamlit as st
import pandas as pd
import joblib

# ================== LOAD MODEL ==================
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
cat_cols = joblib.load("cat_cols.pkl")
num_cols = joblib.load("num_cols.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Clustering Tingkat Obesitas", page_icon="🍎")
st.title("🍎 Clustering Tingkat Obesitas (K-Means)")
st.write(
    "Aplikasi ini mengelompokkan seseorang ke dalam salah satu **cluster** "
    "berdasarkan kebiasaan makan dan gaya hidup, menggunakan model K-Means "
    "yang sudah dilatih pada tahap CRISP-DM."
)

st.header("Masukkan Data Anda")

with st.expander("ℹ️ Keterangan skala FAF, TUE, FCVC, dan NCP"):
    st.markdown("""
- **FCVC (Frekuensi makan sayur):** 1 = tidak pernah, 2 = kadang-kadang, 3 = selalu
- **NCP (Jumlah makan besar per hari):** 1 = 1–2 kali, 2 = 3 kali, 3 = lebih dari 3 kali
- **FAF (Frekuensi aktivitas fisik):** 0 = tidak pernah, 1 = 1–2 hari/minggu, 2 = 2–4 hari/minggu, 3 = 4–5 hari/minggu
- **TUE (Waktu pakai gadget):** 0 = 0–2 jam/hari, 1 = 3–5 jam/hari, 2 = lebih dari 5 jam/hari

Nilai desimal (misalnya 1.5) juga valid, karena mengikuti pola data sintetis pada dataset asli.
""")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 10, 100, 25)
    height = st.number_input("Height (meter)", 1.2, 2.2, 1.65)
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
    family_history = st.selectbox("Riwayat keluarga overweight?", ["yes", "no"])
    favc = st.selectbox("Sering makan tinggi kalori (FAVC)?", ["yes", "no"])
    fcvc = st.slider("Frekuensi makan sayur (FCVC) 1-3", 1.0, 3.0, 2.0)
    ncp = st.slider("Jumlah makan besar per hari (NCP)", 1.0, 4.0, 3.0)
with col2:
    caec = st.selectbox("Ngemil di antara waktu makan (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Merokok?", ["yes", "no"])
    ch2o = st.slider("Konsumsi air/hari (CH2O) liter", 1.0, 3.0, 2.0)
    scc = st.selectbox("Memantau kalori (SCC)?", ["yes", "no"])
    faf = st.slider("Frekuensi aktivitas fisik (FAF)", 0.0, 3.0, 1.0)
    tue = st.slider("Waktu pakai gadget (TUE)", 0.0, 2.0, 1.0)
    calc = st.selectbox("Konsumsi alkohol (CALC)", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Transportasi utama (MTRANS)",
                           ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

if st.button("Prediksi Cluster"):
    input_df = pd.DataFrame([{
        "Gender": gender, "Age": age, "Height": height, "Weight": weight,
        "family_history_with_overweight": family_history, "FAVC": favc,
        "FCVC": fcvc, "NCP": ncp, "CAEC": caec, "SMOKE": smoke,
        "CH2O": ch2o, "SCC": scc, "FAF": faf, "TUE": tue,
        "CALC": calc, "MTRANS": mtrans
    }])

    # Encoding kategorikal dengan encoder yang sama seperti saat training
    input_df[cat_cols] = encoder.transform(input_df[cat_cols])

    # Scaling dengan scaler yang sama seperti saat training (urutan kolom harus sama persis)
    input_scaled = scaler.transform(input_df[feature_columns])

    cluster = kmeans.predict(input_scaled)[0]

    st.success(f"Individu ini masuk ke dalam **Cluster {cluster}**")
    st.write(
        "Catatan: interpretasi tiap cluster (misalnya cenderung berisiko obesitas "
        "tinggi/rendah) mengacu pada hasil profiling cluster di notebook CRISP-DM."
    )