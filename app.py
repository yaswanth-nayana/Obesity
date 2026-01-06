import streamlit as st
import pandas as pd
import pickle
import requests
import io
import numpy as np

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

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

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
                with st.expander("View features"):
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
        
        # Use a form for input
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
            
            # Additional features
            st.subheader("Additional Factors")
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
                tue = st.slider("Technology Use (hours)", 0.0, 2.0, 0.5, 0.1)
            
            mtrans = st.selectbox("Transportation", ["Public_Transportation", "Automobile", "Walking", "Bike", "Motorbike"])
            
            submitted = st.form_submit_button("🔍 Predict Obesity Level", use_container_width=True)
    
    with col2:
        if submitted:
            st.markdown('<h2>📊 Prediction Results</h2>', unsafe_allow_html=True)
            
            try:
                # Prepare input data
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
                
                # Create input dictionary with ALL original feature names
                input_data = {
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
                
                # Get model's expected features - IMPORTANT: Use the exact order from training
                # First, check if model has feature names
                if hasattr(model, 'feature_name_'):
                    # LightGBM model
                    expected_features = list(model.feature_name_)
                elif hasattr(model, 'feature_names_in_'):
                    # scikit-learn model
                    expected_features = list(model.feature_names_in_)
                elif top_features and len(top_features) > 0:
                    # Use top_features from model package
                    expected_features = list(top_features)
                else:
                    # Fallback - try to match with input data keys
                    expected_features = list(input_data.keys())
                
                # DEBUG: Show what features we have
                with st.expander("🔍 Debug Information"):
                    st.write("**Input data keys:**", list(input_data.keys()))
                    st.write("**Expected features:**", expected_features)
                    st.write("**Number of expected features:**", len(expected_features))
                    if scaler:
                        st.write("**Scaler exists:** Yes")
                        if hasattr(scaler, 'feature_names_in_'):
                            st.write("**Scaler features:**", list(scaler.feature_names_in_))
                    else:
                        st.write("**Scaler exists:** No")
                
                # Create a simple array for prediction - most reliable approach
                # Build the feature array in the EXACT order the model expects
                feature_array = []
                
                for feature in expected_features:
                    if feature in input_data:
                        feature_array.append(input_data[feature])
                    else:
                        # If feature is missing, use 0
                        feature_array.append(0)
                
                # Convert to numpy array and reshape for prediction
                feature_array = np.array(feature_array).reshape(1, -1)
                
                # DEBUG: Show the actual values being sent
                with st.expander("🔍 Feature Values Sent"):
                    for i, (feature, value) in enumerate(zip(expected_features, feature_array[0])):
                        st.write(f"{i+1}. {feature}: {value}")
                
                # Try to scale if scaler exists
                if scaler:
                    try:
                        # Check if scaler expects the same number of features
                        if hasattr(scaler, 'n_features_in_'):
                            if scaler.n_features_in_ == len(feature_array[0]):
                                feature_array = scaler.transform(feature_array)
                                st.success("✅ Features scaled successfully")
                            else:
                                st.warning(f"⚠️ Scaler expects {scaler.n_features_in_} features, but we have {len(feature_array[0])}. Using unscaled.")
                        else:
                            # Try to transform anyway
                            feature_array = scaler.transform(feature_array)
                            st.success("✅ Features scaled successfully")
                    except Exception as e:
                        st.warning(f"⚠️ Scaling failed: {str(e)}. Using unscaled features.")
                
                # Make prediction
                with st.spinner("Making prediction..."):
                    try:
                        prediction = model.predict(feature_array)[0]
                        st.success(f"✅ Prediction made successfully")
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
                        # Try with a simple default array
                        default_array = np.zeros((1, len(expected_features)))
                        prediction = model.predict(default_array)[0]
                        st.warning("⚠️ Used default array for prediction")
                
                # DEBUG: Show raw prediction
                with st.expander("🔍 Raw Prediction Info"):
                    st.write(f"**Raw prediction value:** {prediction}")
                    st.write(f"**Prediction type:** {type(prediction)}")
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(feature_array)[0]
                        st.write(f"**Prediction probabilities:**")
                        for i, prob in enumerate(proba):
                            st.write(f"  Class {i}: {prob:.4f}")
                
                # Decode prediction based on your value_counts
                # Your value_counts show this order:
                # Obesity_Type_III (0), Obesity_Type_II (1), Normal_Weight (2), 
                # Obesity_Type_I (3), Insufficient_Weight (4), Overweight_Level_II (5), 
                # Overweight_Level_I (6)
                
                obesity_labels = [
                    "Obesity_Type_III",       # Class 0
                    "Obesity_Type_II",        # Class 1
                    "Normal_Weight",          # Class 2
                    "Obesity_Type_I",         # Class 3
                    "Insufficient_Weight",    # Class 4
                    "Overweight_Level_II",    # Class 5
                    "Overweight_Level_I"      # Class 6
                ]
                
                # Handle prediction value
                if isinstance(prediction, (int, np.integer)):
                    prediction_int = int(prediction)
                else:
                    # Try to convert to int
                    try:
                        prediction_int = int(prediction)
                    except:
                        prediction_int = 5  # Default to Overweight_Level_II
                
                # Try to use label encoder if available
                result = ""
                if 'NObeyesdad' in label_encoders:
                    try:
                        result = label_encoders['NObeyesdad'].inverse_transform([prediction_int])[0]
                    except:
                        if 0 <= prediction_int < len(obesity_labels):
                            result = obesity_labels[prediction_int]
                        else:
                            result = f"Class {prediction_int}"
                else:
                    if 0 <= prediction_int < len(obesity_labels):
                        result = obesity_labels[prediction_int]
                    else:
                        result = f"Class {prediction_int}"
                
                # Store result
                st.session_state.prediction_result = result
                st.session_state.prediction_made = True
                st.session_state.last_prediction_value = prediction_int
                
                # Display result
                prediction_colors = {
                    "Insufficient_Weight": "#3498db",
                    "Normal_Weight": "#2ecc71",
                    "Overweight_Level_I": "#f1c40f",
                    "Overweight_Level_II": "#e67e22",
                    "Obesity_Type_I": "#e74c3c",
                    "Obesity_Type_II": "#c0392b",
                    "Obesity_Type_III": "#7d3c98"
                }
                
                color = prediction_colors.get(result, "#1E3A8A")
                
                st.markdown(f"""
                <div class="prediction-box" style="border-left-color: {color};">
                    <h3>🎯 Predicted Obesity Level</h3>
                    <h1 style="color: {color}; text-align: center; margin: 1rem 0;">{result}</h1>
                    <p style="text-align: center; color: #666;">
                        Class: {prediction_int} | Raw value: {prediction}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show input summary
                st.markdown('<h3>📋 Your Input Summary</h3>', unsafe_allow_html=True)
                
                summary_data = {
                    "BMI": f"{bmi:.1f} ({'Underweight' if bmi < 18.5 else 'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obesity'})",
                    "Age": age,
                    "Weight": f"{weight} kg",
                    "Height": f"{height} m",
                    "Physical Activity": f"{faf}/3",
                    "Vegetable Intake": f"{fcvc}/3",
                    "Family History": "Yes" if family_history == "yes" else "No"
                }
                
                for key, value in summary_data.items():
                    st.write(f"**{key}:** {value}")
                
                # Metrics visualization
                st.markdown('<h3>📈 Key Health Factors</h3>', unsafe_allow_html=True)
                
                metrics = [
                    ("Weight", weight / 100),  # Normalize to 100kg max
                    ("Age", age / 100),        # Normalize to 100 years max
                    ("Physical Activity", faf / 3),
                    ("Vegetable Intake", fcvc / 3),
                    ("Water Intake", ch2o / 3)
                ]
                
                for name, value in metrics:
                    percent = min(int(value * 100), 100)
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
                    </div>
                    """, unsafe_allow_html=True)
                
                # Recommendations
                st.markdown('<h3>📋 Personalized Recommendations</h3>', unsafe_allow_html=True)
                
                recommendations = {
                    "Insufficient_Weight": [
                        "💪 Include strength training 3 times weekly",
                        "🍽️ Eat 5-6 smaller meals throughout day",
                        "🥛 Add protein shakes or supplements",
                        "👨‍⚕️ Consult dietitian for meal plan"
                    ],
                    "Normal_Weight": [
                        "🏃 Maintain 150 mins exercise weekly",
                        "🥗 Continue balanced diet with variety",
                        "⚖️ Monitor weight monthly",
                        "💧 Drink 2+ liters water daily"
                    ],
                    "Overweight_Level_I": [
                        "🚶 Increase daily steps to 10,000",
                        "🥦 Replace processed foods with vegetables",
                        "⏰ Avoid eating 2 hours before bedtime",
                        "📱 Use fitness app to track activity"
                    ],
                    "Overweight_Level_II": [
                        "🏋️ Add 30 mins cardio 5 days/week",
                        "📊 Track calorie intake with food diary",
                        "👨‍⚕️ Schedule check-up with doctor",
                        "🍎 Focus on whole foods, reduce sugar"
                    ],
                    "Obesity_Type_I": [
                        "🏥 Consult doctor for health plan",
                        "🚴 Aim for 60+ mins daily activity",
                        "👨‍🍳 Work with registered dietitian",
                        "🧘 Consider stress management"
                    ],
                    "Obesity_Type_II": [
                        "⚠️ Immediate medical consultation",
                        "📋 Structured weight management program",
                        "🩺 Regular health monitoring",
                        "💊 Discuss medication options"
                    ],
                    "Obesity_Type_III": [
                        "🆘 Urgent medical attention required",
                        "🏨 Multidisciplinary approach needed",
                        "🔍 Surgical options evaluation",
                        "👨‍⚕️ Close medical supervision crucial"
                    ]
                }
                
                rec_list = recommendations.get(result, [
                    "Maintain balanced diet",
                    "Stay physically active",
                    "Consult healthcare professional"
                ])
                
                for i, rec in enumerate(rec_list, 1):
                    st.info(f"{i}. {rec}")
                
                if st.button("🔄 Make Another Prediction", use_container_width=True):
                    st.session_state.prediction_made = False
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                
                # Detailed debug information
                with st.expander("🔧 Technical Details"):
                    st.write("**Error type:**", type(e).__name__)
                    st.write("**Full error:**", str(e))
                    import traceback
                    st.write("**Traceback:**")
                    st.code(traceback.format_exc())
        
        elif st.session_state.prediction_made:
            # Show previous prediction
            st.markdown('<h2>📊 Previous Prediction</h2>', unsafe_allow_html=True)
            
            prediction_label = st.session_state.prediction_result
            
            prediction_colors = {
                "Insufficient_Weight": "#3498db",
                "Normal_Weight": "#2ecc71",
                "Overweight_Level_I": "#f1c40f",
                "Overweight_Level_II": "#e67e22",
                "Obesity_Type_I": "#e74c3c",
                "Obesity_Type_II": "#c0392b",
                "Obesity_Type_III": "#7d3c98"
            }
            
            color = prediction_colors.get(prediction_label, "#1E3A8A")
            
            st.markdown(f"""
            <div class="prediction-box" style="border-left-color: {color};">
                <h3>🎯 Previous Prediction</h3>
                <h1 style="color: {color}; text-align: center; margin: 1rem 0;">{prediction_label}</h1>
                <p style="text-align: center;">
                    Class: {getattr(st.session_state, 'last_prediction_value', 'N/A')}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 Make New Prediction", use_container_width=True):
                st.session_state.prediction_made = False
                st.rerun()
        
        else:
            # Placeholder before prediction
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; background-color: #F8F9FA; border-radius: 10px;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">⚖️</div>
                <h3 style="color: #6B7280;">Ready for Prediction</h3>
                <p>Fill out the form and click <strong>"Predict Obesity Level"</strong></p>
                <p style="color: #10B981; margin-top: 1rem;">
                    <strong>✓ Model loaded successfully</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>⚠️ <strong>Disclaimer:</strong> For educational purposes only. Consult healthcare professionals for medical advice.</p>
    <p>📦 <strong>Model loaded from GitHub:</strong> yaswanth-nayana/Obesity</p>
</div>
""", unsafe_allow_html=True)

