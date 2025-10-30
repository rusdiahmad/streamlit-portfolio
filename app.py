# app.py - Portofolio Streamlit Rusdi Ahmad
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Konfigurasi tampilan halaman
st.set_page_config(page_title="Portofolio Streamlit - Rusdi Ahmad", layout="wide")

# Menu navigasi di sidebar
menu = st.sidebar.radio(
    "Menu Navigasi",
    ["🏠 Beranda", "👤 Tentang Saya", "💼 Proyek Saya", "📊 Visualisasi & Prediksi"]
)

# =============================
# 🏠 BERANDA
# =============================
if menu == "🏠 Beranda":
    st.title("🌐 Portofolio Streamlit — Rusdi Ahmad")
    st.markdown("---")
    st.markdown("""
    Selamat datang di situs portofolio saya 👋  
    Melalui aplikasi ini, Anda dapat mengenal latar belakang saya, melihat beberapa proyek yang pernah saya kerjakan, 
    serta mencoba model prediksi sederhana yang dapat dijalankan langsung di sini.
    """)
    st.info("Gunakan menu di sebelah kiri untuk menjelajahi halaman lainnya ➡️")

# =============================
# 👤 TENTANG SAYA
# =============================
elif menu == "👤 Tentang Saya":
    st.header("👨‍🏫 Tentang Saya")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=160)
    with col2:
        st.subheader("Rusdi Ahmad, M.Sc.")
        st.markdown("""
        - 🎓 **Pendidikan:** Magister Matematika — Universitas Andalas (UNAND)  
        - 🏠 **Domisili:** Kota Bogor, Indonesia  
        - 💼 **Profesi:** Pendidik Matematika & Penggiat Data  
        - 🧠 **Keahlian:** Matematika • Statistika • Machine Learning • Streamlit
        """)
    st.write("📧 Email: rusdiahmad979@gmail.com")
    st.write("🔗 LinkedIn: [linkedin.com/in/rusdi-ahmad-a2948a1a4](https://www.linkedin.com/in/rusdi-ahmad-a2948a1a4)")

    st.markdown("---")
    st.subheader("💡 Motto")
    st.write("*“Matematika adalah bahasa sains — dan Streamlit membuatnya terlihat nyata.”*")

# =============================
# 💼 PROYEK SAYA
# =============================
elif menu == "💼 Proyek Saya":
    st.header("💼 Proyek Saya")
    st.markdown("---")

    projects = [
        {
            "title": "📊 Analisis Nilai UTBK Siswa",
            "desc": "Dashboard interaktif untuk menganalisis distribusi nilai UTBK serta faktor yang memengaruhi keberhasilan siswa.",
            "tags": "Python • Streamlit • Data Visualization"
        },
        {
            "title": "🏨 Prediksi Pembatalan Reservasi Hotel",
            "desc": "Model machine learning untuk memprediksi pembatalan reservasi menggunakan algoritma Random Forest.",
            "tags": "Machine Learning • EDA • Klasifikasi"
        },
        {
            "title": "📘 Analisis Kinerja Akademik Siswa",
            "desc": "Visualisasi data nilai siswa untuk menemukan pola performa dan strategi peningkatan hasil belajar.",
            "tags": "Pendidikan • Analisis Data • Visualisasi"
        }
    ]

    for p in projects:
        st.subheader(p["title"])
        st.write(p["desc"])
        st.caption(p["tags"])
        st.markdown("---")

# =============================
# 📊 VISUALISASI & PREDIKSI
# =============================
elif menu == "📊 Visualisasi & Prediksi":
    st.header("📈 Visualisasi Data & Model Prediksi")
    st.markdown("---")
    st.markdown("""
    Unggah file CSV untuk menampilkan **visualisasi sederhana** serta menjalankan **model prediksi regresi**.
    Jika dataset memiliki kolom `SalePrice`, sistem akan mengenalinya sebagai variabel target.
    """)

    uploaded = st.file_uploader("📤 Unggah file CSV Anda di sini", type=["csv"])
    model_path = Path("trained_model.pkl")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success("✅ File berhasil diunggah!")
        st.subheader("📋 Cuplikan Data")
        st.dataframe(df.head())

        # Visualisasi kolom numerik
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        if numeric_cols:
            st.subheader("📊 Visualisasi Kolom Numerik")
            feature = st.selectbox("Pilih kolom numerik untuk ditampilkan:", numeric_cols)
            fig, ax = plt.subplots()
            ax.hist(df[feature].dropna(), bins=20)
            ax.set_xlabel(feature)
            ax.set_ylabel("Frekuensi")
            ax.set_title(f"Distribusi {feature}")
            st.pyplot(fig)

            # Korelasi
            if st.checkbox("Tampilkan heatmap korelasi"):
                corr = df[numeric_cols].corr()
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                im = ax2.imshow(corr, cmap="coolwarm", interpolation="nearest")
                ax2.set_xticks(range(len(numeric_cols)))
                ax2.set_xticklabels(numeric_cols, rotation=45, ha='right')
                ax2.set_yticks(range(len(numeric_cols)))
                ax2.set_yticklabels(numeric_cols)
                st.pyplot(fig2)
        else:
            st.warning("Tidak ada kolom numerik yang ditemukan dalam dataset.")

        st.markdown("---")
        st.subheader("🤖 Model Machine Learning")

        if "SalePrice" in df.columns:
            if st.button("Latih Model (RandomForest)"):
                X = df.select_dtypes(include=[np.number]).drop(columns=["SalePrice"], errors="ignore").fillna(0)
                y = df["SalePrice"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                rmse = mean_squared_error(y_test, preds, squared=False)
                r2 = r2_score(y_test, preds)
                st.success(f"Model berhasil dilatih! RMSE = {rmse:.2f}, R² = {r2:.3f}")

                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                st.info("Model disimpan sebagai 'trained_model.pkl'")

        if st.button("Jalankan Prediksi (gunakan model yang ada)"):
            if model_path.exists():
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                X_pred = df.select_dtypes(include=[np.number]).fillna(0)
                preds = model.predict(X_pred)
                df["Hasil_Prediksi"] = preds
                st.success("Prediksi berhasil dijalankan!")
                st.dataframe(df.head())
                st.download_button("Unduh Hasil Prediksi", df.to_csv(index=False), "hasil_prediksi.csv")
            else:
                st.warning("Model belum tersedia. Silakan latih model terlebih dahulu.")

    else:
        st.info("Silakan unggah file CSV untuk memulai analisis dan prediksi.")

# =============================
# Footer
# =============================
st.sidebar.write("---")
st.sidebar.caption("📘 Rusdi Ahmad • Portofolio Streamlit 2025")
