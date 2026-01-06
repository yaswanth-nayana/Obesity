import streamlit as st
import pandas as pd
import pickle
import requests
import io

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

# Function to load model from GitHub
def load_model_from_github(model_url):
    """Load model directly from GitHub URL"""
    try:
        # Use raw GitHub URL
        raw_url = model_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
        
        # Download the file
        response = requests.get(raw_url)
        response.raise_for_status()  # Check for errors
        
        # Load pickle from bytes
        model_data = pickle.load(io.BytesIO(response.content))
        return model_data, True, f"Loaded from GitHub: {model_url}"
        
    except Exception as e:
        return None, False, f"Error loading from GitHub: {str(e)}"

# Initialize session state
if 'model_data' not in st.session_state:
    # Your GitHub model URLs
    github_model_urls = [
        "https://github.com/yaswanth-nayana/Obesity/blob/main/model_package.pkl",
        "https://github.com/yaswanth-nayana/Obesity/blob/main/best_model.pkl"
    ]
    
    # Try to load from GitHub
    for model_url in github_model_urls:
        model_data, success, message = load_model_from_github(model_url)
        if success:
            st.session_state.model_data = model_data
            st.session_state.model_source = model_url
            st.session_state.model_message = message
            break
    else:
        st.session_state.model_data = None
        st.session_state.model_message = "Could not load models from GitHub"

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
        st.info(f"**Source:** GitHub Repository")
        
        # Show model type
        if isinstance(st.session_state.model_data, dict) and 'model' in st.session_state.model_data:
            model = st.session_state.model_data['model']
            st.info(f"**Model Type:** {type(model).__name__}")
            
            # Show features if available
            if 'top_features' in st.session_state.model_data and st.session_state.model_data['top_features']:
                st.info(f"**Features:** {len(st.session_state.model_data['top_features'])}")
                with st.expander("View Features"):
                    for i, feature in enumerate(st.session_state.model_data['top_features'], 1):
                        st.write(f"{i}. {feature}")
    else:
        st.error(f"❌ {st.session_state.model_message}")
        st.info("**Tried to load from:**")
        st.write("• https://github.com/yaswanth-nayana/Obesity/blob/main/model_package.pkl")
        st.write("• https://github.com/yaswanth-nayana/Obesity/blob/main/best_model.pkl")
    
    # Reload button
    if st.button("🔄 Reload from GitHub", use_container_width=True):
        # Clear session state and reload
        st.session_state.model_data = None
        st.rerun()

# Main content area
if not st.session_state.model_data:
    # Show model not loaded state
    st.warning("""
    ## ⚠️ Model Not Loaded
    
    The app is trying to load models directly from your GitHub repository.
    
    **Possible issues:**
    1. GitHub URLs might not be accessible
    2. Large file size (models are ~2.4MB)
    3. Network connection issue
    
    **Quick fixes:**
    1. Wait a moment and click **🔄 Reload from GitHub** in sidebar
    2. Make sure your GitHub repository is public
    3. Try uploading the model file manually below
    """)
    
    # Manual upload as fallback
    with st.expander("📤 Upload Model File (Fallback)"):
        uploaded_file = st.file_uploader("Choose a .pkl file", type="pkl")
        if uploaded_file is not None:
            try:
                st.session_state.model_data = pickle.load(uploaded_file)
                st.session_state.model_source = "Uploaded file"
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
                fcvc = st.slider("Vegetable Consumption", 1.0, 3.0, 2.0, 0.1)
            with col_f:
                ncp = st.slider("Number of Main Meals", 1.0, 4.0, 3.0, 0.1)
            
            # Additional features
            st.subheader("Additional Habits")
            col_g, col_h = st.columns(2)
            with col_g:
                caec = st.selectbox("Eating Between Meals", ["no", "Sometimes", "Frequently", "Always"])
            with col_h:
                calc = st.selectbox("Alcohol Consumption", ["no", "Sometimes", "Frequently", "Always"])
            
            col_i, col_j = st.columns(2)
            with col_i:
                smoke = st.selectbox("Smoking", ["yes", "no"])
            with col_j:
                scc = st.selectbox("Calorie Monitoring", ["yes", "no"])
            
            col_k, col_l = st.columns(2)
            with col_k:
                ch2o = st.slider("Water Intake (L)", 0.5, 3.0, 1.5, 0.1)
            with col_l:
                tue = st.slider("Technology Use (hrs)", 0.0, 2.0, 0.5, 0.1)
            
            faf = st.slider("Physical Activity", 0.0, 3.0, 1.0, 0.1)
            
            mtrans = st.selectbox("Transportation", ["Public_Transportation", "Automobile", "Walking", "Bike", "Motorbike"])
            
            submitted = st.form_submit_button("🔍 Predict Obesity Level", use_container_width=True)
    
    with col2:
        if submitted:
            st.markdown('<h2>📊 Prediction Results</h2>', unsafe_allow_html=True)
            
            try:
                # Prepare ALL input data - ALL 16 features in correct order
                # This is the EXACT order the model was trained with
                input_data = {}
                
                # Map categorical values to numeric codes
                gender_map = {"Male": 0, "Female": 1}
                yes_no_map = {"yes": 1, "no": 0}
                caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
                calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
                mtrans_map = {
                    "Public_Transportation": 0, 
                    "Automobile": 1, 
                    "Walking": 2, 
                    "Bike": 3, 
                    "Motorbike": 4
                }
                
                # Create feature dictionary - ALL 16 features in training order
                # Based on the obesity dataset standard features
                feature_dict = {
                    'Gender': gender_map[gender],
                    'Age': float(age),
                    'Height': float(height),
                    'Weight': float(weight),
                    'family_history_with_overweight': yes_no_map[family_history],
                    'FAVC': yes_no_map[favc],
                    'FCVC': float(fcvc),
                    'NCP': float(ncp),
                    'CAEC': caec_map[caec],
                    'SMOKE': yes_no_map[smoke],
                    'CH2O': float(ch2o),
                    'SCC': yes_no_map[scc],
                    'FAF': float(faf),
                    'TUE': float(tue),
                    'CALC': calc_map[calc],
                    'MTRANS': mtrans_map[mtrans]
                }
                
                # Debug: Show what features we're sending
                with st.expander("🔍 Debug: Input Features"):
                    st.write("**Feature Values:**")
                    for key, value in feature_dict.items():
                        st.write(f"{key}: {value}")
                
                # Determine which features to use
                if top_features and len(top_features) > 0:
                    # Use only the top features that the model expects
                    st.info(f"Using {len(top_features)} selected features")
                    selected_features = top_features
                else:
                    # Use all 16 features
                    selected_features = list(feature_dict.keys())
                
                # Create DataFrame with features in the EXACT order
                # This is CRITICAL for matching training order
                df = pd.DataFrame({feature: [feature_dict[feature]] for feature in selected_features})
                
                # Ensure columns are in the right order
                df = df[selected_features]
                
                # Scale if scaler exists
                if scaler:
                    try:
                        df_scaled = scaler.transform(df)
                        st.info("✅ Features scaled successfully")
                    except Exception as scale_error:
                        st.warning(f"⚠️ Scaling skipped: {str(scale_error)}")
                        df_scaled = df.values
                else:
                    df_scaled = df.values
                
                # Debug: Show DataFrame shape and columns
                with st.expander("🔍 Debug: Model Input"):
                    st.write(f"**DataFrame shape:** {df.shape}")
                    st.write(f"**Columns:** {list(df.columns)}")
                    st.write(f"**First row values:** {df.values[0]}")
                
                # Make prediction
                with st.spinner("Making prediction..."):
                    prediction = model.predict(df_scaled)[0]
                
                # Decode prediction
                obesity_labels = [
                    "Insufficient_Weight", 
                    "Normal_Weight", 
                    "Overweight_Level_I",
                    "Overweight_Level_II", 
                    "Obesity_Type_I", 
                    "Obesity_Type_II", 
                    "Obesity_Type_III"
                ]
                
                # Try to decode using label encoder
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
                    <p style="text-align: center; color: #666;">
                        Based on analysis of your lifestyle factors
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Simple metrics visualization
                st.markdown('<h3>📈 Key Health Metrics</h3>', unsafe_allow_html=True)
                
                metrics = [
                    ("BMI", bmi, 40),
                    ("Age", age, 80),
                    ("Physical Activity", faf, 3),
                    ("Vegetable Intake", fcvc, 3),
                    ("Water Intake", ch2o, 3)
                ]
                
                for name, value, max_val in metrics:
                    percent = min(int((value / max_val) * 100), 100)
                    bar_color = "#2ecc71" if percent < 70 else "#f39c12" if percent < 90 else "#e74c3c"
                    
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0;">
                        <div style="display: flex; justify-content: space-between;">
                            <span><strong>{name}</strong></span>
                            <span>{percent}%</span>
                        </div>
                        <div class="feature-bar">
                            <div class="feature-fill" style="width: {percent}%; background-color: {bar_color};"></div>
                        </div>
                        <small>{value} / {max_val}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Simple recommendations
                st.markdown('<h3>📋 Personalized Recommendations</h3>', unsafe_allow_html=True)
                
                if "Insufficient" in result:
                    st.info("1. Increase calorie intake with healthy foods\n2. Include strength training\n3. Eat regular meals")
                elif "Normal" in result:
                    st.success("1. Maintain current lifestyle\n2. Regular exercise\n3. Balanced diet")
                elif "Overweight" in result:
                    st.warning("1. Increase physical activity\n2. Reduce portion sizes\n3. More vegetables")
                else:
                    st.error("1. Consult healthcare professional\n2. Structured exercise plan\n3. Dietary counseling")
                
                # Show feature importance if available
                if hasattr(model, 'feature_importances_') and top_features:
                    with st.expander("🔍 Most Important Features"):
                        for i, (feature, importance) in enumerate(zip(top_features, model.feature_importances_)):
                            importance_percent = int(importance * 1000)  # Scale for visibility
                            st.markdown(f"""
                            <div style="margin: 0.5rem 0;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span>{feature}</span>
                                    <span>{importance_percent}</span>
                                </div>
                                <div class="feature-bar">
                                    <div class="feature-fill" style="width: {min(importance_percent, 100)}%;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                if st.button("🔄 Make Another Prediction", use_container_width=True):
                    st.session_state.prediction_made = False
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                
                # Debug information
                with st.expander("🔧 Technical Details"):
                    st.write("**Error type:**", type(e).__name__)
                    st.write("**Full error:**", str(e))
                    
                    if 'df' in locals():
                        st.write("**DataFrame columns:**", list(df.columns))
                        st.write("**DataFrame shape:**", df.shape)
                    
                    if top_features:
                        st.write("**Expected features (top_features):**", top_features)
                    
                    # Check model's expected features
                    if hasattr(model, 'feature_name_'):
                        st.write("**Model's feature names:**", model.feature_name_)
                    elif hasattr(model, 'feature_importances_'):
                        st.write("**Model has feature importances**")
        
        else:
            # Placeholder before prediction
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; background-color: #F8F9FA; border-radius: 10px;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">⚖️</div>
                <h3 style="color: #6B7280;">Ready for Prediction</h3>
                <p>Fill out the form on the left and click <strong>"Predict Obesity Level"</strong></p>
                <p style="color: #10B981; margin-top: 1rem;">
                    <strong>✓ Model loaded successfully</strong>
                </p>
                <p><small>Features configured: {feature_count}</small></p>
            </div>
            """.format(
                feature_count=len(top_features) if top_features else "All 16"
            ), unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>⚠️ <strong>Disclaimer:</strong> For educational purposes only. Consult healthcare professionals for medical advice.</p>
    <p>📦 <strong>Model loaded from GitHub:</strong> yaswanth-nayana/Obesity</p>
</div>
""", unsafe_allow_html=True)
