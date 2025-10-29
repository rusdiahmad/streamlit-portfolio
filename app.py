# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="My Portfolio with Streamlit", layout="wide")

# -----------------------
# Header / Branding
# -----------------------
st.title("My Portfolio with Streamlit")
st.markdown("""
Selamat datang di portofolio saya. Di sini saya menampilkan profil singkat, beberapa proyek, dan halaman untuk melakukan prediksi menggunakan model ML sederhana.
""")

# -----------------------
# Sidebar navigation
# -----------------------
page = st.sidebar.selectbox("Pilih Halaman", ["Tentang Saya", "Proyek Saya", "Prediksi (Upload CSV)", "Visualisasi & Model"])

# -----------------------
# Tentang Saya
# -----------------------
if page == "Tentang Saya":
    st.header("Tentang Saya")
    st.markdown("""
- **Nama:** Nama Anda
- **Latar Belakang:** S2 Matematika / Data Science / dsb
- **Keahlian:** Python, Machine Learning, Data Visualization, Streamlit
""")
    st.subheader("Kontak")
    st.write("Email: your.email@example.com")
    st.write("LinkedIn: https://www.linkedin.com/in/your-profile")

# -----------------------
# Proyek Saya
# -----------------------
elif page == "Proyek Saya":
    st.header("Proyek Saya")
    projects = [
        {
            "title": "Analisis Harga Rumah (Regression)",
            "desc": "Membangun model prediksi harga rumah, EDA, dan deployment menggunakan Streamlit.",
            "img": None
        },
        {
            "title": "Dashboard Analisis Penjualan",
            "desc": "Dashboard interaktif untuk analisis penjualan dengan filter & visualisasi.",
            "img": None
        },
        {
            "title": "Klasifikasi Kualitas Produk",
            "desc": "Model klasifikasi untuk mengidentifikasi produk berkualitas vs tidak.",
            "img": None
        },
    ]

    for p in projects:
        st.subheader(p["title"])
        st.write(p["desc"])
        if p["img"]:
            st.image(p["img"], use_column_width=True)
        st.markdown("---")

# -----------------------
# Prediksi: Upload CSV dan Trigger
# -----------------------
elif page == "Prediksi (Upload CSV)":
    st.header("Halaman Prediksi")
    st.write("Unggah file CSV berisi fitur yang diperlukan untuk melakukan prediksi harga (atau gunakan contoh).")
    uploaded_file = st.file_uploader("Unggah CSV (header wajib)", type=["csv"])

    model_path = Path("trained_model.pkl")

    # Load model jika ada
    model = None
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            st.success("Model pra-latih ditemukan dan dimuat.")
        except Exception as e:
            st.warning("Gagal memuat model pra-latih: " + str(e))

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview data:")
        st.dataframe(df.head())

        # asumsi kolom target bernama 'SalePrice' (jika ada)
        if "SalePrice" in df.columns:
            st.info("File mengandung kolom 'SalePrice'. Aplikasi ini akan menggunakan kolom ini saat melatih/mengevaluasi model.")
        # Tombol untuk trigger pipeline
        if st.button("Trigger: Train & Predict (jika belum ada model)"):
            with st.spinner("Menjalankan pipeline..."):
                # Jika model belum ada, latih sederhana gunakan kolom numerik saja
                if model is None:
                    st.write("Melatih model RandomForest sederhana dari data yang Anda unggah (menggunakan kolom numerik).")
                    data = df.copy()
                    if "SalePrice" not in data.columns:
                        st.error("Dataset harus berisi kolom target 'SalePrice' untuk pelatihan. Jika file hanya berisi fitur input untuk prediksi, gunakan fitur 'Predict only' di bawah.")
                    else:
                        y = data["SalePrice"].values
                        X = data.select_dtypes(include=[np.number]).drop(columns=["SalePrice"], errors="ignore")
                        X = X.fillna(X.median())
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        rf = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf.fit(X_train, y_train)
                        preds = rf.predict(X_test)
                        rmse = mean_squared_error(y_test, preds, squared=False)
                        mae = mean_absolute_error(y_test, preds)
                        r2 = r2_score(y_test, preds)

                        st.success("Pelatihan selesai.")
                        st.write(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")

                        # Simpan model
                        with open(model_path, "wb") as f:
                            pickle.dump(rf, f)
                        model = rf
                        st.info("Model tersimpan sebagai 'trained_model.pkl' (di repository server).")
                else:
                    st.info("Model sudah ada — langsung dipakai untuk prediksi.")

        st.markdown("### Predict only (pakai model yang sudah ada)")
        if st.button("Jalankan Prediksi pada file ini (pakai model jika ada)"):
            if model is None:
                st.error("Tidak ada model tersedia. Silakan tekan 'Trigger: Train & Predict' dahulu atau upload model pra-latih.")
            else:
                X_pred = df.select_dtypes(include=[np.number]).fillna(0)
                preds = model.predict(X_pred)
                df_out = df.copy()
                df_out["prediction"] = preds
                st.write("Hasil prediksi (preview):")
                st.dataframe(df_out.head())
                st.download_button("Download hasil prediksi (CSV)", df_out.to_csv(index=False), file_name="predictions.csv")

# -----------------------
# Visualisasi & Model
# -----------------------
elif page == "Visualisasi & Model":
    st.header("Visualisasi Data & Performa Model")
    st.write("Anda dapat mengunggah dataset untuk EDA atau melihat performa model yang tersimpan.")

    uploaded = st.file_uploader("Unggah CSV untuk EDA (opsional)", type=["csv"], key="eda")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.subheader("Preview dataset")
        st.dataframe(df.head())

        # Distribusi salah satu fitur numerik (pilih)
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        if numeric_cols:
            col = st.selectbox("Pilih fitur numerik untuk melihat distribusi", numeric_cols)
            bins = st.slider("Jumlah bins (histogram)", 5, 100, 30)
            fig, ax = plt.subplots()
            ax.hist(df[col].dropna(), bins=bins)
            ax.set_title(f"Distribusi {col}")
            st.pyplot(fig)

            # Korelasi (heatmap sederhana)
            if st.checkbox("Tampilkan korelasi (heatmap sederhana)"):
                corr = df[numeric_cols].corr()
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                im = ax2.imshow(corr, interpolation='nearest')
                ax2.set_xticks(range(len(numeric_cols)))
                ax2.set_xticklabels(numeric_cols, rotation=45, ha='right')
                ax2.set_yticks(range(len(numeric_cols)))
                ax2.set_yticklabels(numeric_cols)
                st.pyplot(fig2)
        else:
            st.info("Tidak ditemukan kolom numerik di dataset ini.")

    st.markdown("---")
    st.subheader("Performa model (jika ada)")
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                mdl = pickle.load(f)
            st.success("Model dimuat dari server.")
            st.write("Model: ", type(mdl).__name__)
            st.info("Jika Anda ingin melihat learning curve/metrics, latih model lokal dengan data kemudian upload file log/metrics atau modifikasi kode sesuai kebutuhan.")
        except Exception as e:
            st.error("Gagal memuat model: " + str(e))
    else:
        st.info("Belum ada model tersimpan. Gunakan halaman 'Prediksi' untuk melatih dan menyimpan model sederhana.")

# Footer
st.sidebar.write("Guidance: Untuk tugas, sertakan link GitHub repo dan link deploy Streamlit pada LMS.")
