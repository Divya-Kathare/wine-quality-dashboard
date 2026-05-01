# ===================== IMPORTS =====================
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Wine Quality App", layout="wide")

# ===================== LIGHT THEME =====================
st.markdown("""
<style>
body {background-color: #ffffff;}
.main {background-color: #ffffff;}
</style>
""", unsafe_allow_html=True)

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    df = pd.read_csv("wine_data.csv")

    # 🔥 DO NOT LOWERCASE (fixes pH issue)
    df.columns = df.columns.str.strip()

    # Remove leakage column
    if 'good' in df.columns:
        df = df.drop('good', axis=1)

    return df

df = load_data()

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    return joblib.load("wine_model.pkl")

model = load_model()

# ===================== HEADER =====================
st.markdown("""
<h1 style='text-align:center;'>🍷 Wine Quality Analysis Dashboard</h1>
<p style='text-align:center;'>Explore patterns in wine chemistry and quality</p>
""", unsafe_allow_html=True)


# ===================== TABS =====================
tabs = st.tabs(["📊 Overview", "📈 Analysis", "💡 Insights", "🤖 Prediction"])


# ===================== OVERVIEW =====================
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", len(df))
    col2.metric("Avg Alcohol", round(df["alcohol"].mean(), 2))
    col3.metric("Avg pH", round(df["pH"].mean(), 2))
    col4.metric("Avg Quality", round(df["quality"].mean(), 2))

    st.subheader("📊 Dataset Preview")
    st.dataframe(df)

# ===================== ANALYSIS =====================
with tabs[1]:

    st.subheader("📊 Data Analysis")

    # ================= EDA TYPE FILTER =================
    eda_type = st.selectbox("Select EDA Type", ["Univariate", "Bivariate"])

    var_type = st.selectbox("Select Variable Type", ["Numerical", "Categorical"])

    # ================= UNIVARIATE =================
    if eda_type == "Univariate":

        if var_type == "Numerical":
            num_cols = df.select_dtypes(include=['int64','float64']).columns
            col = st.selectbox("Select Numerical Column", num_cols)

            fig = px.histogram(df, x=col, title=f"{col} Distribution")
            st.plotly_chart(fig, use_container_width=True)

        else:
            cat_cols = df.select_dtypes(include=['object']).columns
            col = st.selectbox("Select Categorical Column", cat_cols)

            c1, c2 = st.columns(2)

            with c1:
                fig1 = px.bar(df[col].value_counts(), title=f"{col} Count")
                st.plotly_chart(fig1, use_container_width=True)

            with c2:
                fig2 = px.pie(df, names=col, title=f"{col} Share")
                st.plotly_chart(fig2, use_container_width=True)

    # ================= BIVARIATE =================
    else:

        if var_type == "Numerical":
            num_cols = df.select_dtypes(include=['int64','float64']).columns

            col1, col2 = st.columns(2)
            x_col = col1.selectbox("Select X", num_cols)
            y_col = col2.selectbox("Select Y", num_cols)

            fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)

        else:
            cat_cols = df.select_dtypes(include=['object']).columns
            num_cols = df.select_dtypes(include=['int64','float64']).columns

            col1, col2 = st.columns(2)
            cat = col1.selectbox("Select Category", cat_cols)
            num = col2.selectbox("Select Value", num_cols)

            fig = px.box(df, x=cat, y=num, title=f"{num} vs {cat}")
            st.plotly_chart(fig, use_container_width=True)

    # ================= CORRELATION HEATMAP =================
    st.subheader("🔗 Correlation Heatmap")

    numeric_df = df.select_dtypes(include=['int64','float64'])

    fig_corr = px.imshow(
        numeric_df.corr(),
        text_auto=True,
        aspect="auto"
    )

    fig_corr.update_layout(
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    st.plotly_chart(fig_corr, use_container_width=True)

# ===================== INSIGHTS =====================
with tabs[2]:

    st.subheader("💡 Key Insights")

    st.markdown("""
1. Alcohol strongly increases wine quality  
2. Volatile acidity reduces quality  
3. Sulphates improve stability  
4. Balanced pH is important  
5. Density affects body  

---

### 📌 Recommendations
- Optimize alcohol levels  
- Control acidity carefully  
- Maintain sulphates  
- Monitor pH levels  
""")

# ===================== PREDICTION =====================
with tabs[3]:

    st.subheader("Predict Wine Quality")

    input_data = {}

    # ✅ UNIQUE KEY (fix duplicate error)
    wine_type = st.selectbox("Wine Type", ["Red", "White"], key="wine_type_pred")

    input_data["color"] = 0 if wine_type == "Red" else 1

    # Feature columns
    feature_cols = df.drop("quality", axis=1).columns

    cols = st.columns(3)

    for i, col in enumerate(feature_cols):

        if col == "color":
            continue

        with cols[i % 3]:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())

            input_data[col] = st.slider(
                col,
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                key=f"slider_{col}"   # ✅ unique keys
            )

    # ✅ BUTTON INSIDE TAB
    if st.button("Predict Quality", key="predict_btn"):

        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_cols]

        prediction = model.predict(input_df)[0]

        st.metric("Predicted Quality", round(prediction, 2))

        # ---------------- RESULT ----------------
        if prediction >= 7:
            st.success("🍾 Excellent Wine")
        elif prediction >= 6:
            st.info("👍 Good Wine")
        elif prediction >= 5:
            st.warning("⚖️ Average Wine")
        else:
            st.error("❌ Low Quality Wine")

        # ---------------- INSIGHTS ----------------
        st.subheader("📊 Prediction Insights")

        avg = df.mean(numeric_only=True)

        insights = []

        if input_data["alcohol"] > avg["alcohol"] + 0.5:
            insights.append("High alcohol content is boosting quality")

        if input_data["volatile acidity"] > avg["volatile acidity"] + 0.05:
            insights.append("High volatile acidity is hurting wine quality")

        if input_data["sulphates"] < avg["sulphates"] - 0.05:
            insights.append("Low sulphates are reducing stability")

        if input_data["density"] > avg["density"]:
            insights.append("High density affects balance")

        if input_data["pH"] > avg["pH"]:
            insights.append("Higher pH weakens acidity")

        if input_data["citric acid"] < avg["citric acid"]:
            insights.append("Low citric acid reduces freshness")

        if not insights:
            insights.append("Wine is balanced and close to optimal")

        for ins in insights:
            st.info(ins)

        # ---------------- RECOMMENDATIONS ----------------
        st.subheader("🎯 How to Improve This Wine")

        recommendations = []

        if prediction < 6:

            if input_data["alcohol"] < avg["alcohol"]:
                recommendations.append("Increase alcohol slightly")

            if input_data["volatile acidity"] > avg["volatile acidity"]:
                recommendations.append("Reduce volatile acidity")

            if input_data["sulphates"] < avg["sulphates"]:
                recommendations.append("Increase sulphates")

            if input_data["citric acid"] < avg["citric acid"]:
                recommendations.append("Increase citric acid")

            if input_data["pH"] > avg["pH"]:
                recommendations.append("Lower pH")

            if input_data["density"] > avg["density"]:
                recommendations.append("Reduce density")

            if not recommendations:
                recommendations = [
                    "Fine-tune alcohol and acidity balance",
                    "Adjust sulphates slightly"
                ]

        else:
            recommendations = [
                "Wine is already good quality",
                "Minor refinements can improve taste"
            ]

        for rec in recommendations:
            if prediction >= 6:
                st.success(rec)
            else:
                st.warning(rec)