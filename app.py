# app.py (Final Revised Version)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Konfigurasi halaman
st.set_page_config(page_title="Rusdi Ahmad Portfolio", layout="wide")

# Sidebar navigasi
menu = st.sidebar.radio(
    "Navigation Menu",
    ["🏠 Home", "👤 About Me", "💼 My Projects", "📊 Visualization & Model"]
)

# =============================
# 🏠 1. HOME PAGE
# =============================
if menu == "🏠 Home":
    st.title("🌐 Rusdi Ahmad — Streamlit Portfolio Project")
    st.markdown("---")
    st.markdown("""
    ### 👋 Welcome!
    This web application was created as part of the **Streamlit Portfolio Assignment**.
 
    """)


# =============================
# 👤 2. ABOUT ME
# =============================
elif menu == "👤 About Me":
    st.header("👨‍🏫 About Me")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=160)
    with col2:
        st.subheader("Rusdi Ahmad, M.Sc.")
        st.markdown("""
        - 🎓 **Education:** Master of Mathematics — Universitas Andalas (UNAND)
        - 🏠 **Location:** Bogor, Indonesia
        - 💼 **Profession:** Mathematics Educator & Data Science Enthusiast
        - 🧠 **Skills:** Mathematics • Data Science • Machine Learning • Streamlit
        """)
    st.write("📧 Email: rusdiahmad979@gmail.com")
    st.write("🔗 LinkedIn: [linkedin.com/in/rusdi-ahmad-a2948a1a4](https://www.linkedin.com/in/rusdi-ahmad-a2948a1a4)")

    st.markdown("---")
    st.subheader("💡 Motto")
    st.write("*“Mathematics is the language of science — Streamlit makes it visible.”*")

# =============================
# 💼 3. MY PROJECTS
# =============================
elif menu == "💼 My Projects":
    st.header("💼 My Projects")
    st.markdown("---")

    projects = [
        {
            "title": "📊 UTBK Score Analysis Dashboard",
            "desc": "An interactive dashboard that explores UTBK student data to analyze performance patterns and success factors.",
            "tags": "Python • Streamlit • Data Visualization"
        },
        {
            "title": "🏨 Hotel Booking Cancellation Prediction",
            "desc": "A simple machine learning model to predict hotel booking cancellations using RandomForest.",
            "tags": "Machine Learning • EDA • Classification"
        },
        {
            "title": "📘 Student Performance Analytics",
            "desc": "A visual analysis of student grades to identify key improvement areas in learning outcomes.",
            "tags": "Education • Analytics • Data Storytelling"
        }
    ]

    for p in projects:
        st.subheader(p["title"])
        st.write(p["desc"])
        st.caption(p["tags"])
        st.markdown("---")

# =============================
# 📊 4. VISUALIZATION & MODEL
# =============================
elif menu == "📊 Visualization & Model":
    st.header("📈 Data Visualization & Prediction Model")
    st.markdown("---")
    st.markdown("""
    Upload your CSV dataset below to visualize its distribution and, if applicable, train or test a simple prediction model.
    """)

    uploaded = st.file_uploader("📤 Upload CSV file", type=["csv"])
    model_path = Path("trained_model.pkl")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success("✅ File uploaded successfully!")
        st.subheader("📋 Data Preview")
        st.dataframe(df.head())

        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        if numeric_cols:
            st.subheader("📊 Numeric Feature Visualization")

            # Histogram
            feature = st.selectbox("Select a numeric column to visualize:", numeric_cols)
            fig, ax = plt.subplots()
            ax.hist(df[feature].dropna(), bins=20)
            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {feature}")
            st.pyplot(fig)

            # Korelasi
            if st.checkbox("Show correlation heatmap"):
                corr = df[numeric_cols].corr()
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                im = ax2.imshow(corr, cmap="coolwarm", interpolation="nearest")
                ax2.set_xticks(range(len(numeric_cols)))
                ax2.set_xticklabels(numeric_cols, rotation=45, ha='right')
                ax2.set_yticks(range(len(numeric_cols)))
                ax2.set_yticklabels(numeric_cols)
                st.pyplot(fig2)
        else:
            st.warning("No numeric columns found in this dataset.")

        st.markdown("---")
        st.subheader("🧮 Train or Use Prediction Model")

        if "SalePrice" in df.columns:
            if st.button("Train Model (RandomForest)"):
                X = df.select_dtypes(include=[np.number]).drop(columns=["SalePrice"], errors="ignore").fillna(0)
                y = df["SalePrice"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                rmse = mean_squared_error(y_test, preds, squared=False)
                r2 = r2_score(y_test, preds)
                st.success(f"Model trained successfully! RMSE = {rmse:.2f}, R² = {r2:.3f}")

                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                st.info("Model saved as 'trained_model.pkl'")

        if st.button("Run Prediction using existing model"):
            if model_path.exists():
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                X_pred = df.select_dtypes(include=[np.number]).fillna(0)
                preds = model.predict(X_pred)
                df["Prediction"] = preds
                st.success("Prediction completed!")
                st.dataframe(df.head())
                st.download_button("Download Results", df.to_csv(index=False), "predictions.csv")
            else:
                st.warning("No model found. Please train a model first.")

    else:
        st.info("Please upload a CSV file to start visualization or prediction.")

# =============================
# Footer
# =============================
st.sidebar.write("---")
st.sidebar.caption("📘 Created by Rusdi Ahmad — Streamlit Portfolio Project 2025")
