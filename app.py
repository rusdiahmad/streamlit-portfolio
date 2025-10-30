# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# Konfigurasi halaman
st.set_page_config(page_title="Rusdi Ahmad Portfolio", layout="wide")

# Sidebar navigasi
menu = st.sidebar.radio(
    "Navigation Menu",
    ["🏠 Home", "👤 About Me", "💼 My Projects", "📈 Visualization & Prediction"]
)

# =============================
# 🏠 1. HOME PAGE
# =============================
if menu == "🏠 Home":
    st.title("Welcome to My Streamlit Portfolio 🌐")
    st.write("---")
    st.markdown("""
    ### 👋 Hello Everyone!
    This web app is created as part of my **Streamlit Portfolio Project**.
    Here, you can explore my background, view my data projects, and even run your own prediction using a simple ML model.  
    Aplikasi ini dibuat untuk memenuhi **tugas pembuatan portofolio Streamlit**, sesuai petunjuk pada modul tugas.
    """)
    st.info("Use the sidebar to navigate through the pages 👉")

# =============================
# 👤 2. ABOUT ME
# =============================
elif menu == "👤 About Me":
    st.header("👨‍🏫 About Me")
    st.write("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=180)
    with col2:
        st.subheader("Rusdi Ahmad, M.Si.")
        st.markdown("""
        - 🎓 **Education:** Master of Mathematics (UNAND)  
        - 📍 **Location:** Bogor, Indonesia  
        - 💼 **Occupation:** Mathematics Educator & Data Enthusiast  
        - 🧠 **Expertise:** Mathematics, Statistics, Data Science, and Machine Learning  
        """)
    st.write("✉️ Email: rusdiahmad979@gmail.com")
    st.write("🔗 LinkedIn: [linkedin.com/in/rusdi-ahmad-a2948a1a4](https://www.linkedin.com/in/rusdi-ahmad-a2948a1a4)")

    st.write("---")
    st.subheader("🧩 Skills & Tools")
    st.markdown("""
    - Python (Pandas, NumPy, Scikit-learn)
    - Data Visualization (Matplotlib, Streamlit)
    - Statistical Analysis & Modeling
    - Education & Mathematical Problem Solving
    """)

# =============================
# 💼 3. MY PROJECTS
# =============================
elif menu == "💼 My Projects":
    st.header("💼 Featured Projects")
    st.write("---")

    projects = [
        {
            "title": "1️⃣ UTBK Score Analysis Dashboard",
            "desc": "Interactive dashboard analyzing UTBK student data to explore score distributions and success factors.",
            "tags": "Python • Streamlit • Data Analysis"
        },
        {
            "title": "2️⃣ Hotel Booking Cancellation Prediction",
            "desc": "Predicting hotel booking cancellations using RandomForest — includes EDA, preprocessing, and model evaluation.",
            "tags": "Machine Learning • EDA • Classification"
        },
        {
            "title": "3️⃣ Student Performance Analytics",
            "desc": "Visualizing student academic data to identify performance trends and improvement strategies.",
            "tags": "Data Visualization • Education • Analytics"
        }
    ]

    for p in projects:
        st.subheader(p["title"])
        st.write(p["desc"])
        st.caption(p["tags"])
        st.markdown("---")

# =============================
# 📈 4. VISUALIZATION & PREDICTION
# =============================
elif menu == "📈 Visualization & Prediction":
    st.header("📊 Data Visualization & Prediction Model")
    st.write("---")

    uploaded = st.file_uploader("📤 Upload your CSV file here", type=["csv"])
    model_path = Path("simple_model.pkl")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success("✅ File successfully uploaded!")
        st.subheader("Preview of Data")
        st.dataframe(df.head())

        # Visualisasi distribusi kolom numerik
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        if numeric_cols:
            st.subheader("📊 Feature Distribution")
            selected_col = st.selectbox("Select a numeric column to visualize:", numeric_cols)
            fig, ax = plt.subplots()
            ax.hist(df[selected_col].dropna(), bins=20)
            ax.set_title(f"Distribution of {selected_col}")
            st.pyplot(fig)

        # Korelasi sederhana
        if st.checkbox("Show correlation heatmap"):
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(corr, cmap="coolwarm", interpolation="nearest")
            ax.set_xticks(range(len(numeric_cols)))
            ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
            ax.set_yticks(range(len(numeric_cols)))
            ax.set_yticklabels(numeric_cols)
            st.pyplot(fig)

        # Pelatihan model sederhana
        if "SalePrice" in df.columns:
            st.subheader("🧮 Train Simple Regression Model")
            if st.button("Train RandomForest Model"):
                X = df.select_dtypes(include=[np.number]).drop(columns=["SalePrice"], errors="ignore").fillna(0)
                y = df["SalePrice"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                rmse = mean_squared_error(y_test, preds, squared=False)
                r2 = r2_score(y_test, preds)
                st.success(f"Model trained successfully! RMSE = {rmse:.2f}, R² = {r2:.3f}")

                # Save model
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                st.info("Model saved to simple_model.pkl")

        # Prediksi (jika model sudah ada)
        if st.button("Run Prediction using existing model"):
            if model_path.exists():
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                X_pred = df.select_dtypes(include=[np.number]).fillna(0)
                preds = model.predict(X_pred)
                df["Prediction"] = preds
                st.success("Prediction completed!")
                st.dataframe(df.head())
                st.download_button("Download Prediction Result", df.to_csv(index=False), "predictions.csv")
            else:
                st.warning("No trained model found. Please train model first.")

    else:
        st.info("Please upload a CSV file to start analysis or prediction.")

# =============================
# Footer
# =============================
st.sidebar.write("---")
st.sidebar.caption("Created by Rusdi Ahmad • Streamlit Portfolio Project 2025")

