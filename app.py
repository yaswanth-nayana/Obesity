import streamlit as st
import pandas as pd
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Obesity Level Predictor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #F0F8FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E3A8A;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #1E3A8A;
    }
    .feature-bar {
        height: 20px;
        background-color: #E5E7EB;
        border-radius: 10px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .feature-fill {
        height: 100%;
        background-color: #1E3A8A;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Simple model loading function
def load_model_file(filename):
    """Load a model from a pickle file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            return model_data, True, f"Loaded from {filename}"
        return None, False, f"File {filename} not found"
    except Exception as e:
        return None, False, f"Error loading {filename}: {str(e)}"

# Initialize session state
if 'model_data' not in st.session_state:
    # Try to load model_package.pkl first, then best_model.pkl
    for model_file in ['model_package.pkl', 'best_model.pkl']:
        model_data, success, message = load_model_file(model_file)
        if success:
            st.session_state.model_data = model_data
            st.session_state.model_file = model_file
            st.session_state.model_message = message
            break
    else:
        st.session_state.model_data = None
        st.session_state.model_message = "No model files found (tried model_package.pkl and best_model.pkl)"

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# App title and description
st.markdown('<h1 class="main-header">⚖️ Obesity Level Prediction System</h1>', unsafe_allow_html=True)
st.markdown("This application uses machine learning to predict obesity levels based on lifestyle and dietary habits.")

# Sidebar for model information
with st.sidebar:
    st.header("📦 Model Status")
    
    if st.session_state.model_data:
        st.success(f"✅ Model Loaded")
        st.info(f"**Source:** {st.session_state.model_file}")
        
        # Show model type
        if isinstance(st.session_state.model_data, dict) and 'model' in st.session_state.model_data:
            model = st.session_state.model_data['model']
            st.info(f"**Model Type:** {type(model).__name__}")
            
            # Show features if available
            if 'top_features' in st.session_state.model_data and st.session_state.model_data['top_features']:
                st.info(f"**Features:** {len(st.session_state.model_data['top_features'])}")
    else:
        st.error(f"❌ {st.session_state.model_message}")
        st.info("**Expected files:** model_package.pkl or best_model.pkl in app directory")
        
        # Show current directory files
        st.subheader("📁 Current Files")
        files = [f for f in os.listdir('.') if f.endswith('.pkl')]
        if files:
            for file in files:
                size = os.path.getsize(file)
                st.write(f"• {file} ({size:,} bytes)")
        else:
            st.write("No .pkl files found")
    
    # Manual reload button
    if st.button("🔄 Reload Models", use_container_width=True):
        for model_file in ['model_package.pkl', 'best_model.pkl']:
            model_data, success, message = load_model_file(model_file)
            if success:
                st.session_state.model_data = model_data
                st.session_state.model_file = model_file
                st.session_state.model_message = message
                st.rerun()
                break

# Main content area
if not st.session_state.model_data:
    # Show model not loaded state
    st.warning("""
    ## ⚠️ Model Not Loaded
    
    Place one of these files in your app directory:
    
    1. **model_package.pkl** (preferred)
    2. **best_model.pkl**
    
    Then click **🔄 Reload Models** in the sidebar.
    """)
    
    # Quick file upload option
    with st.expander("📤 Upload Model File (Alternative)"):
        uploaded_file = st.file_uploader("Choose a .pkl file", type="pkl")
        if uploaded_file is not None:
            try:
                st.session_state.model_data = pickle.load(uploaded_file)
                st.session_state.model_file = "Uploaded file"
                st.success("✅ Model uploaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load: {str(e)}")

else:
    # Model is loaded - show prediction interface
    model_data = st.session_state.model_data
    
    # Extract model and components
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
        scaler = model_data.get('scaler')
        top_features = model_data.get('top_features', [])
        label_encoders = model_data.get('label_encoders', {})
    else:
        # Assume it's a raw model
        model = model_data
        scaler = None
        top_features = []
        label_encoders = {}
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2>📝 Input Your Information</h2>', unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            # Personal Information
            st.subheader("Personal Info")
            col_a, col_b = st.columns(2)
            with col_a:
                gender = st.selectbox("Gender", ["Male", "Female"])
            with col_b:
                age = st.number_input("Age", 1, 120, 30)
            
            col_c, col_d = st.columns(2)
            with col_c:
                height = st.number_input("Height (m)", 0.5, 2.5, 1.7, 0.01, format="%.2f")
            with col_d:
                weight = st.number_input("Weight (kg)", 20.0, 300.0, 70.0, 0.1, format="%.1f")
            
            # Calculate BMI
            bmi = weight / (height ** 2)
            st.markdown(f"""
            <div class="metric-card">
                <h4>Your BMI</h4>
                <h2>{bmi:.1f}</h2>
                <p><strong>Category:</strong> {'Underweight' if bmi < 18.5 else 'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obesity'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Lifestyle Factors
            st.subheader("Lifestyle")
            family_history = st.selectbox("Family Overweight History", ["yes", "no"])
            favc = st.selectbox("High Calorie Food", ["yes", "no"])
            
            col_e, col_f = st.columns(2)
            with col_e:
                fcvc = st.slider("Vegetables", 1.0, 3.0, 2.0, 0.1)
            with col_f:
                ncp = st.slider("Main Meals", 1.0, 4.0, 3.0, 0.1)
            
            faf = st.slider("Physical Activity", 0.0, 3.0, 1.0, 0.1)
            
            submitted = st.form_submit_button("🔍 Predict Obesity Level", use_container_width=True)
    
    with col2:
        if submitted:
            st.markdown('<h2>📊 Prediction Results</h2>', unsafe_allow_html=True)
            
            # Prepare input data
            input_data = {
                'Gender': 0 if gender == "Male" else 1,
                'Age': age,
                'Height': height,
                'Weight': weight,
                'family_history_with_overweight': 1 if family_history == "yes" else 0,
                'FAVC': 1 if favc == "yes" else 0,
                'FCVC': fcvc,
                'NCP': ncp,
                'FAF': faf
            }
            
            # Add default values for missing features
            default_features = {
                'CAEC': 0, 'SMOKE': 0, 'CH2O': 1.5, 'SCC': 0, 
                'TUE': 0.5, 'CALC': 0, 'MTRANS': 0
            }
            input_data.update(default_features)
            
            try:
                # Create DataFrame
                df = pd.DataFrame([input_data])
                
                # Scale if scaler exists
                if scaler:
                    df_scaled = scaler.transform(df)
                else:
                    df_scaled = df.values
                
                # Select features if specified
                if top_features and len(top_features) > 0:
                    # Get available features
                    available_features = [f for f in top_features if f in df.columns]
                    if available_features:
                        X = df_scaled[:, [list(df.columns).index(f) for f in available_features]]
                    else:
                        X = df_scaled
                else:
                    X = df_scaled
                
                # Make prediction
                prediction = model.predict(X)[0]
                
                # Decode prediction
                obesity_labels = [
                    "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I",
                    "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"
                ]
                
                if 'NObeyesdad' in label_encoders:
                    try:
                        result = label_encoders['NObeyesdad'].inverse_transform([prediction])[0]
                    except:
                        result = obesity_labels[prediction] if prediction < len(obesity_labels) else f"Class {prediction}"
                else:
                    result = obesity_labels[prediction] if prediction < len(obesity_labels) else f"Class {prediction}"
                
                # Display result
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>🎯 Predicted Obesity Level</h3>
                    <h1 style="text-align: center; color: #1E3A8A;">{result}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Simple metrics visualization
                st.markdown('<h3>📈 Key Factors</h3>', unsafe_allow_html=True)
                
                metrics = [
                    ("Weight", weight / 150),
                    ("Age", age / 80),
                    ("Physical Activity", faf / 3),
                    ("Vegetable Intake", fcvc / 3)
                ]
                
                for name, value in metrics:
                    percent = min(int(value * 100), 100)
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0;">
                        <div style="display: flex; justify-content: space-between;">
                            <span><strong>{name}</strong></span>
                            <span>{percent}%</span>
                        </div>
                        <div class="feature-bar">
                            <div class="feature-fill" style="width: {percent}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Simple recommendations
                st.markdown('<h3>📋 Recommendations</h3>', unsafe_allow_html=True)
                
                if "Insufficient" in result:
                    st.info("1. Increase calorie intake with healthy foods\n2. Include strength training\n3. Eat regular meals")
                elif "Normal" in result:
                    st.success("1. Maintain current lifestyle\n2. Regular exercise\n3. Balanced diet")
                elif "Overweight" in result:
                    st.warning("1. Increase physical activity\n2. Reduce portion sizes\n3. More vegetables")
                else:
                    st.error("1. Consult healthcare professional\n2. Structured exercise plan\n3. Dietary counseling")
                
                if st.button("🔄 New Prediction", use_container_width=True):
                    st.session_state.prediction_made = False
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
        
        else:
            # Placeholder before prediction
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; background-color: #F8F9FA; border-radius: 10px;">
                <h3 style="color: #6B7280;">👈 Enter Information</h3>
                <p>Fill out the form and click "Predict" to see results.</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>⚠️ <strong>Disclaimer:</strong> For educational purposes only. Consult healthcare professionals for medical advice.</p>
</div>
""", unsafe_allow_html=True)
