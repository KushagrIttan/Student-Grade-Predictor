import streamlit as st
import pandas as pd
import pickle
import shap
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Student Grade Predictor", layout="wide")

# --- DATA AND MODEL LOADING ---
@st.cache_data
def load_data():
    if os.path.exists('student-mat.csv'):
        return pd.read_csv('student-mat.csv', sep=';')
    return None

@st.cache_resource
def load_pipeline():
    if os.path.exists('student_model.pkl'):
        with open('student_model.pkl', 'rb') as f:
            return pickle.load(f)
    return None

data = load_data()
pipeline = load_pipeline()

# --- UI STYLING ---
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# --- SHAP EXPLANATION FUNCTION ---
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# --- PAGE 1: PREDICTOR ---
def predictor_page():
    st.title("🧠 Advanced Grade Predictor")
    st.markdown("Use the sliders and dropdowns to input student data and see the model's prediction.")

    if pipeline is None or data is None:
        st.error("Model or data not found. Please run the training script and ensure 'student-mat.csv' is present.")
        return

    st.markdown("---")

    # Create columns for inputs
    col1, col2 = st.columns(2)

    # --- INPUT WIDGETS ---
    with col1:
        st.header("Academic & Personal Info")
        G1 = st.slider("First Period Grade (G1)", 0, 20, 15)
        G2 = st.slider("Second Period Grade (G2)", 0, 20, 16)
        studytime = st.select_slider("Weekly Study Time", options=[1, 2, 3, 4], format_func=lambda x: {1: '<2 hrs', 2: '2-5 hrs', 3: '5-10 hrs', 4: '>10 hrs'}[x])
        failures = st.slider("Number of Past Failures", 0, 4, 0)
        absences = st.slider("Number of School Absences", 0, 93, 5)
        higher = st.selectbox("Wants to take higher education?", data['higher'].unique(), format_func=lambda x: x.capitalize())
        internet = st.selectbox("Has internet access at home?", data['internet'].unique(), format_func=lambda x: x.capitalize())
        
    with col2:
        st.header("Demographic & Family Info")
        sex = st.selectbox("Sex", data['sex'].unique())
        age = st.slider("Age", 15, 22, 16)
        address = st.selectbox("Address Type", data['address'].unique(), format_func=lambda x: {'U': 'Urban', 'R': 'Rural'}[x])
        Mjob = st.selectbox("Mother's Job", data['Mjob'].unique())
        Fjob = st.selectbox("Father's Job", data['Fjob'].unique())
        famsup = st.selectbox("Family educational support?", data['famsup'].unique(), format_func=lambda x: x.capitalize())
        paid = st.selectbox("Extra paid classes?", data['paid'].unique(), format_func=lambda x: x.capitalize())

    # Create a dictionary of all other features with default values (median or mode)
    # This is to ensure the model receives all expected 32 features
    default_features = {
        'school': 'GP', 'famsize': 'GT3', 'Pstatus': 'A', 'Medu': 3, 'Fedu': 3,
        'reason': 'course', 'guardian': 'mother', 'traveltime': 1, 'schoolsup': 'no',
        'activities': 'no', 'nursery': 'yes', 'romantic': 'no', 'famrel': 4,
        'freetime': 3, 'goout': 3, 'Dalc': 1, 'Walc': 2, 'health': 3
    }
    
    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        **default_features,
        'sex': sex, 'age': age, 'address': address, 'Mjob': Mjob, 'Fjob': Fjob,
        'studytime': studytime, 'failures': failures, 'absences': absences,
        'higher': higher, 'internet': internet, 'G1': G1, 'G2': G2,
        'famsup': famsup, 'paid': paid
    }, index=[0])


    st.markdown("---")

    if st.button("🚀 Predict Final Grade", key="predict_button"):
        prediction = pipeline.predict(input_data)[0]
        
        st.markdown(
            f"""
            <div class="prediction-box">
                <p class="prediction-text">Predicted Final Grade (G3): {prediction:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.header("💡 Prediction Explanation")
        st.markdown("The chart below shows which features pushed the prediction higher (in green) or lower (in red).")

        # SHAP Explanation
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['regressor']
        
        # Get feature names after one-hot encoding
        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(input_features=preprocessor.transformers_[1][2])
        all_feature_names = preprocessor.transformers_[0][2] + list(cat_feature_names)

        # Transform the single input row
        input_transformed = preprocessor.transform(input_data)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_transformed)

        # Create SHAP force plot for the single prediction
        force_plot = shap.force_plot(explainer.expected_value, shap_values[0, :], 
                                     feature_names=all_feature_names, 
                                     matplotlib=False)
        st_shap(force_plot, 400)


# --- PAGE 2: DATA EXPLORER ---
def explorer_page():
    st.title("📊 Data Explorer")
    st.markdown("View the original dataset used to train the model.")
    if data is not None:
        st.dataframe(data)
    else:
        st.error("Dataset not found.")

# --- MAIN APP ---
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Predictor", "Data Explorer"])

    if page == "Predictor":
        predictor_page()
    elif page == "Data Explorer":
        explorer_page()

if __name__ == "__main__":
    main()