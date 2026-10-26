import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Student Academic Risk Monitoring",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Predictive Student Academic Risk Monitoring")
st.write("A student academic risk prediction application.")

DATA_FILE = "student_performance.csv"
TARGET = "Academic_Status"

FEATURES = [
    "Attendance_Rate",
    "Quiz_Average",
    "Assignment_Average",
    "Absences",
    "Late_Submissions",
    "Study_Hours",
    "Previous_GPA",
    "Assessment_Average",
    "Academic_Concern_Count"
]

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

required = FEATURES + [TARGET]
missing = [col for col in required if col not in df.columns]

if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

df = df[required].copy()
df = df.dropna(subset=[TARGET])

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), FEATURES)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

tab1, tab2, tab3 = st.tabs([
    "Dashboard",
    "Predict Student Risk",
    "Dataset"
])

with tab1:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(df))
    c2.metric("At Risk", int((y == "At Risk").sum()))
    c3.metric("Satisfactory", int((y == "Satisfactory").sum()))

    st.subheader("Model Evaluation")
    st.metric("Test Accuracy", f"{accuracy:.2%}")
    st.caption(
        "This app trains a new model from the CSV. "
        "Replace this with the original fitted midterm model before final submission."
    )

    st.subheader("Academic Status Distribution")
    st.bar_chart(y.value_counts())

with tab2:
    st.subheader("Enter Student Information")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        attendance = c1.number_input(
            "Attendance Rate (%)", min_value=0.0, max_value=100.0, value=85.0
        )
        quiz = c2.number_input(
            "Quiz Average", min_value=0.0, max_value=100.0, value=80.0
        )
        assignment = c3.number_input(
            "Assignment Average", min_value=0.0, max_value=100.0, value=80.0
        )

        absences = c1.number_input(
            "Absences", min_value=0, max_value=300, value=3
        )
        late = c2.number_input(
            "Late Submissions", min_value=0, max_value=300, value=2
        )
        study = c3.number_input(
            "Study Hours", min_value=0.0, max_value=24.0, value=2.0
        )

        gpa = c1.number_input(
            "Previous GPA", min_value=0.0, max_value=100.0, value=85.0
        )
        assessment = c2.number_input(
            "Assessment Average", min_value=0.0, max_value=100.0, value=80.0
        )
        concerns = c3.number_input(
            "Academic Concern Count", min_value=0, max_value=20, value=1
        )

        submitted = st.form_submit_button("Predict Academic Status")

    if submitted:
        student = pd.DataFrame([{
            "Attendance_Rate": attendance,
            "Quiz_Average": quiz,
            "Assignment_Average": assignment,
            "Absences": absences,
            "Late_Submissions": late,
            "Study_Hours": study,
            "Previous_GPA": gpa,
            "Assessment_Average": assessment,
            "Academic_Concern_Count": concerns
        }])

        result = model.predict(student)[0]
        st.subheader("Prediction Result")
        if result == "At Risk":
            st.warning("Prediction: At Risk")
        else:
            st.success("Prediction: Satisfactory")

with tab3:
    st.subheader("Student Dataset")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download Dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="student_performance.csv",
        mime="text/csv"
    )

st.caption("BS Information Systems | Systems Analysis and Design")
