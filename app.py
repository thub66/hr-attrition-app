import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="👥",
    layout="wide",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("attrition_pipeline.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("HR-Employee-Attrition.csv")
    return df

pipeline = load_model()
df_raw   = load_data()

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Predict Attrition", "📊 EDA Dashboard", "📈 Model Performance"])

# =============================================================================
# HOME
# =============================================================================
if page == "🏠 Home":
    st.title("👥 HR Employee Attrition Prediction")
    st.markdown("""
    ### 🧠 Problem Statement
    Employee attrition is a critical issue for organizations — it drives up hiring costs,
    reduces productivity, and lowers team morale.

    This app uses a **Machine Learning classification model** trained on IBM HR data to
    predict whether an employee is likely to leave the company.

    ---
    ### 🎯 Key Features Influencing Attrition
    | Feature | Impact |
    |---|---|
    | Monthly Income | Lower income → higher attrition |
    | Years at Company | Fewer years → higher attrition |
    | Job Role | Certain roles show higher turnover |
    | Overtime | Overtime workers leave more |
    | Distance from Home | Farther commute → higher attrition |
    | Job Satisfaction | Lower satisfaction → higher attrition |

    ---
    ### 🚀 How to Use
    - **Predict Attrition** — Enter employee details and get an instant prediction
    - **EDA Dashboard** — Explore dataset patterns visually
    - **Model Performance** — View how each model scored
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", len(df_raw))
    col2.metric("Attrition Rate", f"{(df_raw['Attrition']=='Yes').mean()*100:.1f}%")
    col3.metric("Features Used", "34")

# =============================================================================
# PREDICT
# =============================================================================
elif page == "🔍 Predict Attrition":
    st.title("🔍 Predict Employee Attrition")
    st.markdown("Fill in the employee details below and click **Predict**.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👤 Personal Info")
            age              = st.slider("Age", 18, 60, 35)
            gender           = st.selectbox("Gender", ["Male", "Female"])
            marital_status   = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            distance_home    = st.slider("Distance From Home (km)", 1, 30, 5)
            education        = st.selectbox("Education Level", [1, 2, 3, 4, 5],
                                             format_func=lambda x: {1:"Below College",2:"College",3:"Bachelor",4:"Master",5:"Doctor"}[x])
            education_field  = st.selectbox("Education Field",
                                             ["Life Sciences","Medical","Marketing","Technical Degree","Human Resources","Other"])

        with col2:
            st.subheader("💼 Job Details")
            department       = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            job_role         = st.selectbox("Job Role",
                                             ["Sales Executive","Research Scientist","Laboratory Technician",
                                              "Manufacturing Director","Healthcare Representative","Manager",
                                              "Sales Representative","Research Director","Human Resources"])
            job_level        = st.selectbox("Job Level", [1, 2, 3, 4, 5])
            job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4],
                                             format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
            job_involvement  = st.selectbox("Job Involvement", [1, 2, 3, 4],
                                             format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
            business_travel  = st.selectbox("Business Travel",
                                             ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
            overtime         = st.selectbox("Overtime", ["Yes", "No"])

        with col3:
            st.subheader("📊 Work History & Compensation")
            monthly_income       = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=500)
            daily_rate           = st.number_input("Daily Rate", 100, 1500, 800)
            hourly_rate          = st.number_input("Hourly Rate", 30, 100, 65)
            monthly_rate         = st.number_input("Monthly Rate", 2000, 27000, 14000)
            percent_salary_hike  = st.slider("Percent Salary Hike", 11, 25, 15)
            stock_option         = st.selectbox("Stock Option Level", [0, 1, 2, 3])
            total_working_years  = st.slider("Total Working Years", 0, 40, 8)
            years_at_company     = st.slider("Years at Company", 0, 40, 5)
            years_in_role        = st.slider("Years in Current Role", 0, 18, 3)
            years_since_promo    = st.slider("Years Since Last Promotion", 0, 15, 1)
            years_with_manager   = st.slider("Years with Current Manager", 0, 17, 3)
            training_times       = st.slider("Training Times Last Year", 0, 6, 2)
            num_companies_worked = st.slider("Num Companies Worked", 0, 9, 2)
            env_satisfaction     = st.selectbox("Environment Satisfaction", [1,2,3,4],
                                                 format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
            relationship_sat     = st.selectbox("Relationship Satisfaction", [1,2,3,4],
                                                 format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
            work_life_balance    = st.selectbox("Work Life Balance", [1,2,3,4],
                                                 format_func=lambda x: {1:"Bad",2:"Good",3:"Better",4:"Best"}[x])
            performance_rating   = st.selectbox("Performance Rating", [3, 4],
                                                 format_func=lambda x: {3:"Excellent",4:"Outstanding"}[x])

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        # Build raw input row matching original dataset columns
        input_dict = {
            "Age": age,
            "BusinessTravel": business_travel,
            "DailyRate": daily_rate,
            "Department": department,
            "DistanceFromHome": distance_home,
            "Education": education,
            "EducationField": education_field,
            "EmployeeCount": 1,
            "EmployeeNumber": 1,
            "EnvironmentSatisfaction": env_satisfaction,
            "Gender": gender,
            "HourlyRate": hourly_rate,
            "JobInvolvement": job_involvement,
            "JobLevel": job_level,
            "JobRole": job_role,
            "JobSatisfaction": job_satisfaction,
            "MaritalStatus": marital_status,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies_worked,
            "Over18": "Y",
            "OverTime": overtime,
            "PercentSalaryHike": percent_salary_hike,
            "PerformanceRating": performance_rating,
            "RelationshipSatisfaction": relationship_sat,
            "StandardHours": 80,
            "StockOptionLevel": stock_option,
            "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times,
            "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_role,
            "YearsSinceLastPromotion": years_since_promo,
            "YearsWithCurrManager": years_with_manager,
        }

        input_df = pd.DataFrame([input_dict])

        # Apply same preprocessing as training
        # 1. IQR capping on numeric cols
        num_cols = input_df.select_dtypes(include=["int64", "float64"]).columns
        df_ref = df_raw.copy()
        le_ref = LabelEncoder()
        df_ref["Attrition"] = le_ref.fit_transform(df_ref["Attrition"])
        ref_num = df_ref.select_dtypes(include=["int64", "float64"]).columns
        for col in num_cols:
            if col in ref_num:
                Q1 = df_ref[col].quantile(0.25)
                Q3 = df_ref[col].quantile(0.75)
                IQR = Q3 - Q1
                input_df[col] = np.clip(input_df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

        # 2. Label encode Attrition-like binary cols — not needed for input
        # 3. One-Hot Encode categorical features
        cat_cols = input_df.select_dtypes(include="object").columns.tolist()
        input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

        # Align columns with training data
        df_proc = df_raw.copy()
        df_proc["Attrition"] = LabelEncoder().fit_transform(df_proc["Attrition"])
        ref_num2 = df_proc.select_dtypes(include=["int64","float64"]).columns
        for col in df_proc.select_dtypes(include=["int64","float64"]).columns:
            Q1 = df_proc[col].quantile(0.25); Q3 = df_proc[col].quantile(0.75)
            IQR = Q3 - Q1
            df_proc[col] = np.clip(df_proc[col], Q1-1.5*IQR, Q3+1.5*IQR)
        cat_f = df_proc.select_dtypes(include="object").columns.tolist()
        df_proc = pd.get_dummies(df_proc, columns=cat_f, drop_first=True)
        X_ref = df_proc.drop("Attrition", axis=1)

        # Reindex to match training columns
        input_encoded = input_encoded.reindex(columns=X_ref.columns, fill_value=0)

        prediction = pipeline.predict(input_encoded)[0]
        proba      = pipeline.predict_proba(input_encoded)[0]

        st.divider()
        if prediction == 1:
            st.error(f"### ⚠️ High Attrition Risk — {proba[1]*100:.1f}% probability of leaving")
            st.markdown("""
            **Recommended Actions:**
            - Review compensation package
            - Assess job satisfaction and workload
            - Offer career development opportunities
            - Consider flexible work arrangements
            """)
        else:
            st.success(f"### ✅ Low Attrition Risk — {proba[0]*100:.1f}% probability of staying")
            st.markdown("This employee is likely to remain with the organization.")

        col1, col2 = st.columns(2)
        col1.metric("Stay Probability",   f"{proba[0]*100:.1f}%")
        col2.metric("Leave Probability",  f"{proba[1]*100:.1f}%")

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(["Stay", "Leave"], [proba[0], proba[1]], color=["#2ecc71", "#e74c3c"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Prediction Probability")
        st.pyplot(fig)

# =============================================================================
# EDA DASHBOARD
# =============================================================================
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Overview", "Attrition Analysis", "Correlations"])

    with tab1:
        st.subheader("Dataset Overview")
        st.dataframe(df_raw.head(10), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Shape:**", df_raw.shape)
            st.write("**Attrition Distribution:**")
            st.dataframe(df_raw["Attrition"].value_counts())
        with col2:
            fig, ax = plt.subplots()
            df_raw["Attrition"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax,
                                                         colors=["#2ecc71","#e74c3c"])
            ax.set_ylabel("")
            ax.set_title("Attrition Distribution")
            st.pyplot(fig)

    with tab2:
        st.subheader("Attrition by Feature")
        cat_options = ["Department", "JobRole", "Gender", "MaritalStatus",
                       "BusinessTravel", "OverTime", "EducationField"]
        selected_cat = st.selectbox("Select Categorical Feature", cat_options)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(x=selected_cat, hue="Attrition", data=df_raw, ax=ax)
        plt.xticks(rotation=45, ha="right")
        ax.set_title(f"{selected_cat} vs Attrition")
        st.pyplot(fig)

        st.divider()
        num_options = ["Age", "MonthlyIncome", "YearsAtCompany",
                       "DistanceFromHome", "TotalWorkingYears", "JobSatisfaction"]
        selected_num = st.selectbox("Select Numerical Feature", num_options)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.boxplot(x="Attrition", y=selected_num, data=df_raw, ax=ax2,
                    palette={"Yes": "#e74c3c", "No": "#2ecc71"})
        ax2.set_title(f"{selected_num} vs Attrition")
        st.pyplot(fig2)

    with tab3:
        st.subheader("Correlation Heatmap")
        df_corr = df_raw.copy()
        df_corr["Attrition"] = LabelEncoder().fit_transform(df_corr["Attrition"])
        num_df = df_corr.select_dtypes(include=["int64","float64"])
        fig3, ax3 = plt.subplots(figsize=(14, 10))
        sns.heatmap(num_df.corr(), annot=True, fmt=".1f", cmap="coolwarm",
                    ax=ax3, linewidths=0.5)
        ax3.set_title("Correlation Heatmap")
        st.pyplot(fig3)

# =============================================================================
# MODEL PERFORMANCE
# =============================================================================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance Comparison")

    results = pd.DataFrame({
        "Model":     ["Logistic Regression","Decision Tree","Random Forest",
                      "Gradient Boosting","XGBoost","KNN","SVM"],
        "Accuracy":  [0.76190, 0.80952, 0.87075, 0.88095, 0.87075, 0.64286, 0.87415],
        "Precision": [0.29870, 0.27027, 0.52941, 0.57143, 0.52000, 0.21053, 0.53125],
        "Recall":    [0.58974, 0.25641, 0.23077, 0.41026, 0.33333, 0.61538, 0.43590],
        "F1 Score":  [0.39655, 0.26316, 0.32143, 0.47761, 0.40625, 0.31373, 0.47887],
    })

    st.dataframe(results.sort_values("Accuracy", ascending=False)
                         .style.highlight_max(axis=0, color="#d4edda"),
                 use_container_width=True)

    metric = st.selectbox("Select Metric to Visualize", ["Accuracy","Precision","Recall","F1 Score"])
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if v == results[metric].max() else "#3498db" for v in results[metric]]
    ax.barh(results["Model"], results[metric], color=colors)
    ax.set_xlabel(metric)
    ax.set_title(f"Model Comparison — {metric}")
    ax.axvline(results[metric].max(), color="red", linestyle="--", alpha=0.5, label="Best")
    ax.legend()
    st.pyplot(fig)

    st.info("""
    **🏆 Best Model: Gradient Boosting** — Highest accuracy (88.1%) and precision (57.1%)

    **Balanced Choice: SVM** — Best F1-score (0.479) balancing precision and recall

    **Best Recall: KNN** — Catches the most actual attrition cases (61.5% recall)
    """)
