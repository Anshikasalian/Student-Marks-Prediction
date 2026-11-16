
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


st.set_page_config(
    page_title="Student Marks Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Prediction")

st.write(
    "Enter your academic information below "
    "to predict your Mathematics marks."
)


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "notebook" / "lin_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "notebook" / "preprocessor.pkl"
DATASET_PATH = BASE_DIR / "data" / "exam_marks.csv"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


try:
    model, preprocessor = load_model()
    df = load_dataset()

except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()

except Exception as e:
    st.error(f"Error loading saved files: {e}")
    st.stop()

TARGET_COLUMN = "math score"

if TARGET_COLUMN not in df.columns:
    st.error(f"Target column '{TARGET_COLUMN}' not found.")
    st.stop()

X = df.drop(columns=[TARGET_COLUMN])

feature_columns = X.columns.tolist()

numeric_features = X.select_dtypes(
    exclude="object"
).columns.tolist()

categorical_features = X.select_dtypes(
    include="object"
).columns.tolist()

st.divider()

st.subheader("📝 Student Details")

st.write("Please fill in all the required information.")

student_values = {}

with st.form("prediction_form"):

    for feature in feature_columns:

        if feature in numeric_features:

            minimum = float(X[feature].min())
            maximum = float(X[feature].max())
            default = float(X[feature].median())

            value = st.number_input(
                feature.replace("_", " ").title(),
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=1.0
            )

            student_values[feature] = value

        elif feature in categorical_features:

            options = sorted(
                X[feature].dropna().astype(str).unique().tolist()
            )

            if options:
                value = st.selectbox(
                    feature.replace("_", " ").title(),
                    options
                )

                student_values[feature] = value

    submitted = st.form_submit_button(
        "🔮 Predict My Marks",
        use_container_width=True
    )

if submitted:

    try:
        student_df = pd.DataFrame(
            [student_values],
            columns=feature_columns
        )

        processed_student = preprocessor.transform(student_df)

        prediction = model.predict(processed_student)[0]

        st.divider()

        st.subheader("🎯 Prediction Result")

        st.metric(
            label="Predicted Mathematics Marks",
            value=f"{prediction:.2f}"
        )

        if prediction >= 90:
            st.success("Excellent! Outstanding predicted performance!")

        elif prediction >= 75:
            st.success("Great job! Your predicted performance is good.")

        elif prediction >= 50:
            st.info("Good effort! There is room for improvement.")

        else:
            st.warning("Keep practising and work towards improving your marks.")

        st.caption(
            "Note: This is an estimated prediction based on "
            "historical student performance data. "
            "Actual examination marks may vary."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
