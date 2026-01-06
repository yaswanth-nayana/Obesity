import streamlit as st
import pandas as pd
import pickle
import requests
import io
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 2rem;
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
            st.session_state.model_loaded = True
            break
    else:
        st.session_state.model_data = None
        st.session_state.model_message = "Could not load models from GitHub"
        st.session_state.model_loaded = False

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# App title and description
st.markdown('<h1 class="main-header">⚖️ Obesity Level Prediction System</h1>', unsafe_allow_html=True)
st.markdown("This application uses machine learning to predict obesity levels based on lifestyle and dietary habits.")

# Sidebar for model information
with st.sidebar:
    st.header("🔧 Model Configuration")
    
    if st.session_state.model_loaded:
        st.success("✅ Model loaded from GitHub!")
        if st.session_state.model_message:
            st.info(f"*{st.session_state.model_message}*")
    else:
        st.error("❌ Model not loaded")
        if st.session_state.model_message:
            st.error(f"*{st.session_state.model_message}*")
    
    # Manual reload option
    if st.button("🔄 Try Reload from GitHub", use_container_width=True):
        # Clear and reload
        for key in ['model_data', 'model_source', 'model_message', 'model_loaded']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    # File selector for manual upload
    st.subheader("📤 Upload Model File (Alternative)")
    uploaded_file = st.file_uploader("Choose a .pkl file", type="pkl", label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            model_data = pickle.load(uploaded_file)
            
            if isinstance(model_data, dict) and 'model' in model_data:
                st.session_state.model_data = model_data
                st.session_state.model_loaded = True
                st.session_state.model_source = "Uploaded file"
                st.success("✅ Model uploaded successfully!")
                st.rerun()
            elif hasattr(model_data, 'predict'):  # It's a scikit-learn model
                st.session_state.model_data = {
                    'model': model_data,
                    'scaler': None,
                    'top_features': None,
                    'label_encoders': {}
                }
                st.session_state.model_loaded = True
                st.session_state.model_source = "Uploaded file"
                st.success("✅ Model uploaded successfully!")
                st.rerun()
            else:
                st.error("Invalid model format!")
                
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
    
    st.divider()
    
    # Model information section
    if st.session_state.model_loaded:
        model_data = st.session_state.model_data
        st.header("📊 Model Information")
        
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            st.info(f"**Model Type:** {type(model).__name__}")
            
            if model_data.get('top_features'):
                st.info(f"**Features Used:** {len(model_data['top_features'])}")
                with st.expander("View Selected Features"):
                    for i, feature in enumerate(model_data['top_features'], 1):
                        st.write(f"{i}. {feature}")
            
            if model_data.get('scaler'):
                st.info(f"**Scaler:** {type(model_data['scaler']).__name__}")
            
            if model_data.get('label_encoders'):
                st.info(f"**Label Encoders:** {len(model_data['label_encoders'])}")
        else:
            st.info(f"**Model Type:** {type(model_data).__name__}")

# Main content area
if not st.session_state.model_loaded:
    # Show model not loaded state
    st.warning("""
    ## ⚠️ Model Not Loaded
    
    The app is trying to load models directly from your GitHub repository.
    
    **Quick fixes:**
    1. Click **🔄 Try Reload from GitHub** in sidebar
    2. Make sure your GitHub repository is public
    3. Upload model file manually using the sidebar uploader
    
    **Tried to load from:**
    - https://github.com/yaswanth-nayana/Obesity/blob/main/model_package.pkl
    - https://github.com/yaswanth-nayana/Obesity/blob/main/best_model.pkl
    """)

else:
    # Model is loaded - show the prediction interface
    model_data = st.session_state.model_data
    
    # Extract model and components
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
        scaler = model_data.get('scaler')
        top_features = model_data.get('top_features')
        label_encoders = model_data.get('label_encoders', {})
    else:
        # Assume it's a raw model
        model = model_data
        scaler = None
        top_features = None
        label_encoders = {}
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📝 Input Your Information</h2>', unsafe_allow_html=True)
        
        # Create input form
        with st.form("prediction_form"):
            # Group similar features
            st.subheader("Personal Information")
            
            # Demographics
            col_a, col_b = st.columns(2)
            with col_a:
                gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender")
            with col_b:
                age = st.number_input("Age", 1, 120, 30, help="Enter your age in years")
            
            # Physical measurements
            col_c, col_d = st.columns(2)
            with col_c:
                height = st.number_input("Height (meters)", 0.5, 2.5, 1.7, 0.01, format="%.2f", 
                                       help="Enter your height in meters")
            with col_d:
                weight = st.number_input("Weight (kg)", 20.0, 300.0, 70.0, 0.1, format="%.1f", 
                                       help="Enter your weight in kilograms")
            
            # Calculate BMI in real-time
            bmi = weight / (height ** 2)
            bmi_category = (
                'Underweight' if bmi < 18.5 else
                'Normal weight' if bmi < 25 else
                'Overweight' if bmi < 30 else
                'Obesity Class I' if bmi < 35 else
                'Obesity Class II' if bmi < 40 else
                'Obesity Class III'
            )
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Your BMI</h4>
                <h2>{bmi:.1f}</h2>
                <p><strong>Category:</strong> {bmi_category}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            st.subheader("Family History & Habits")
            
            # Family history and habits
            family_history = st.selectbox("Family History with Overweight", ["yes", "no"], 
                                         help="Does your family have a history of overweight?")
            favc = st.selectbox("Frequent high calorie food", ["yes", "no"], 
                               help="Do you frequently eat high calorie food?")
            
            st.divider()
            st.subheader("Dietary Habits")
            
            # Dietary habits in columns
            col_e, col_f = st.columns(2)
            with col_e:
                fcvc = st.slider("Vegetable consumption", 1.0, 3.0, 2.0, 0.1,
                                help="How often do you eat vegetables? (1=Never, 3=Always)")
            with col_f:
                ncp = st.slider("Number of main meals", 1.0, 4.0, 3.0, 0.1,
                               help="How many main meals do you have daily?")
            
            # More dietary habits
            col_g, col_h = st.columns(2)
            with col_g:
                caec = st.selectbox("Eating between meals", ["no", "Sometimes", "Frequently", "Always"],
                                   help="Do you eat between meals?")
            with col_h:
                calc = st.selectbox("Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"],
                                   help="How often do you consume alcohol?")
            
            st.divider()
            st.subheader("Physical Activity & Lifestyle")
            
            # Physical activity
            faf = st.slider("Physical activity frequency", 0.0, 3.0, 1.0, 0.1,
                           help="How often are you physically active? (0=Never, 3=Very Often)")
            
            # Transportation
            mtrans = st.selectbox("Transportation used", 
                                 ["Public_Transportation", "Automobile", "Walking", "Bike", "Motorbike"],
                                 help="What is your primary mode of transportation?")
            
            # Additional lifestyle factors
            col_i, col_j = st.columns(2)
            with col_i:
                smoke = st.selectbox("Smoking habit", ["yes", "no"], help="Do you smoke?")
            with col_j:
                scc = st.selectbox("Calorie monitoring", ["yes", "no"], 
                                  help="Do you monitor your calorie intake?")
            
            # Water consumption and technology use
            col_k, col_l = st.columns(2)
            with col_k:
                ch2o = st.slider("Daily water consumption (liters)", 0.5, 3.0, 1.5, 0.1,
                                help="How much water do you drink daily?")
            with col_l:
                tue = st.slider("Technology use (hours)", 0.0, 2.0, 0.5, 0.1,
                               help="Daily hours using electronic devices")
            
            # Submit button
            submitted = st.form_submit_button("🔍 Predict Obesity Level", type="primary", use_container_width=True)
    
    with col2:
        if submitted:
            st.markdown('<h2 class="sub-header">📊 Prediction Results</h2>', unsafe_allow_html=True)
            
            # Store user data
            user_data = {
                'Gender': gender,
                'Age': age,
                'Height': height,
                'Weight': weight,
                'family_history_with_overweight': family_history,
                'FAVC': favc,
                'FCVC': fcvc,
                'NCP': ncp,
                'CAEC': caec,
                'SMOKE': smoke,
                'CH2O': ch2o,
                'SCC': scc,
                'FAF': faf,
                'TUE': tue,
                'CALC': calc,
                'MTRANS': mtrans
            }
            
            st.session_state.user_data = user_data
            
            # Display BMI calculation
            st.markdown(f"""
            <div class="prediction-box">
                <h3>📈 Your BMI: {bmi:.1f}</h3>
                <p><strong>BMI Category:</strong> {bmi_category}</p>
                <p><strong>Height:</strong> {height}m | <strong>Weight:</strong> {weight}kg</p>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                # Prepare data for prediction - convert categorical to numeric
                prediction_data = {}
                
                # Encoding mappings
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
                
                # Apply encoding
                prediction_data['Gender'] = gender_map[gender]
                prediction_data['Age'] = float(age)
                prediction_data['Height'] = float(height)
                prediction_data['Weight'] = float(weight)
                prediction_data['family_history_with_overweight'] = yes_no_map[family_history]
                prediction_data['FAVC'] = yes_no_map[favc]
                prediction_data['FCVC'] = float(fcvc)
                prediction_data['NCP'] = float(ncp)
                prediction_data['CAEC'] = caec_map[caec]
                prediction_data['SMOKE'] = yes_no_map[smoke]
                prediction_data['CH2O'] = float(ch2o)
                prediction_data['SCC'] = yes_no_map[scc]
                prediction_data['FAF'] = float(faf)
                prediction_data['TUE'] = float(tue)
                prediction_data['CALC'] = calc_map[calc]
                prediction_data['MTRANS'] = mtrans_map[mtrans]
                
                # Define ALL original features in correct order
                all_original_features = [
                    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
                    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
                    'CALC', 'MTRANS'
                ]
                
                # Determine which features to use
                if top_features and len(top_features) > 0:
                    # Use only top features
                    selected_features = [f for f in top_features if f in all_original_features]
                    if not selected_features:
                        selected_features = all_original_features
                else:
                    # Use all features
                    selected_features = all_original_features
                
                # Create DataFrame with selected features in correct order
                full_df = pd.DataFrame(columns=selected_features)
                
                # Fill the DataFrame with our data
                for feature in selected_features:
                    if feature in prediction_data:
                        full_df[feature] = [prediction_data[feature]]
                    else:
                        # Default values if feature missing (shouldn't happen)
                        if feature in ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']:
                            full_df[feature] = [0.0]
                        else:
                            full_df[feature] = [0]
                
                # Debug: Show the features being used
                with st.expander("🔍 Debug: Model Input"):
                    st.write("**Selected features:**", selected_features)
                    st.write("**DataFrame shape:**", full_df.shape)
                    st.write("**DataFrame columns:**", list(full_df.columns))
                    st.write("**Data values:**", full_df.values[0])
                
                # Scale features if scaler exists
                if scaler:
                    try:
                        scaled_data = scaler.transform(full_df)
                    except Exception as scale_error:
                        st.warning(f"⚠️ Scaling skipped: {str(scale_error)}")
                        scaled_data = full_df.values
                else:
                    scaled_data = full_df.values
                
                # Make prediction
                with st.spinner("Making prediction..."):
                    prediction = model.predict(scaled_data)[0]
                
                # Decode prediction
                prediction_label = f"Class {prediction}"
                if 'NObeyesdad' in label_encoders:
                    try:
                        prediction_label = label_encoders['NObeyesdad'].inverse_transform([prediction])[0]
                    except:
                        # Use default mapping if inverse transform fails
                        obesity_classes = {
                            0: "Insufficient_Weight",
                            1: "Normal_Weight",
                            2: "Overweight_Level_I",
                            3: "Overweight_Level_II",
                            4: "Obesity_Type_I",
                            5: "Obesity_Type_II",
                            6: "Obesity_Type_III"
                        }
                        prediction_label = obesity_classes.get(prediction, f"Class {prediction}")
                else:
                    # Fallback if no label encoder
                    obesity_classes = {
                        0: "Insufficient_Weight",
                        1: "Normal_Weight",
                        2: "Overweight_Level_I",
                        3: "Overweight_Level_II",
                        4: "Obesity_Type_I",
                        5: "Obesity_Type_II",
                        6: "Obesity_Type_III"
                    }
                    prediction_label = obesity_classes.get(prediction, f"Class {prediction}")
                
                # Store result
                st.session_state.prediction_result = prediction_label
                st.session_state.prediction_made = True
                
                # Display prediction with color coding
                prediction_colors = {
                    "Insufficient_Weight": "#FF6B6B",
                    "Normal_Weight": "#4ECDC4",
                    "Overweight_Level_I": "#FFD166",
                    "Overweight_Level_II": "#FF9A76",
                    "Obesity_Type_I": "#FF6B6B",
                    "Obesity_Type_II": "#FF4757",
                    "Obesity_Type_III": "#FF3838"
                }
                
                color = prediction_colors.get(prediction_label, "#1E3A8A")
                
                st.markdown(f"""
                <div class="prediction-box" style="border-left-color: {color};">
                    <h3 style="color: {color};">🎯 Predicted Obesity Level</h3>
                    <h1 style="color: {color}; text-align: center; margin: 1rem 0;">{prediction_label}</h1>
                    <p style="text-align: center;">
                        Based on your input data, the model predicts this obesity category.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show feature importance visualization
                st.markdown('<h3 class="sub-header">📊 Key Metrics</h3>', unsafe_allow_html=True)
                
                # Create simple bar visualization using HTML/CSS
                important_metrics = {
                    'Weight': min(weight / 150, 1.0),  # Normalize to 0-1
                    'Age': min(age / 80, 1.0),
                    'Physical Activity': faf / 3.0,
                    'Vegetable Consumption': fcvc / 3.0,
                    'Water Intake': ch2o / 3.0,
                    'Technology Use': tue / 2.0
                }
                
                for metric, value in important_metrics.items():
                    percentage = min(int(value * 100), 100)
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0;">
                        <div style="display: flex; justify-content: space-between;">
                            <span><strong>{metric}</strong></span>
                            <span>{percentage}%</span>
                        </div>
                        <div class="feature-bar">
                            <div class="feature-fill" style="width: {percentage}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Recommendations based on prediction
                st.markdown('<h3 class="sub-header">📋 Personalized Recommendations</h3>', unsafe_allow_html=True)
                
                recommendations = {
                    "Insufficient_Weight": [
                        "Increase calorie intake with nutrient-dense foods",
                        "Incorporate strength training to build muscle mass",
                        "Have regular, balanced meals throughout the day",
                        "Consider consulting a nutritionist for a personalized plan"
                    ],
                    "Normal_Weight": [
                        "Maintain your current healthy lifestyle",
                        "Continue regular physical activity",
                        "Monitor portion sizes to maintain weight",
                        "Stay hydrated and eat a balanced diet"
                    ],
                    "Overweight_Level_I": [
                        "Increase daily physical activity by 30 minutes",
                        "Reduce portion sizes and avoid late-night eating",
                        "Increase vegetable and fruit consumption",
                        "Limit high-calorie beverages"
                    ],
                    "Overweight_Level_II": [
                        "Aim for 45-60 minutes of daily physical activity",
                        "Track your food intake using a journal or app",
                        "Consult with a healthcare professional",
                        "Focus on whole foods and reduce processed foods"
                    ],
                    "Obesity_Type_I": [
                        "Consult with a doctor for a comprehensive plan",
                        "Aim for 60+ minutes of daily physical activity",
                        "Work with a registered dietitian",
                        "Consider behavioral therapy for eating habits"
                    ],
                    "Obesity_Type_II": [
                        "Immediate consultation with healthcare provider",
                        "Structured weight management program recommended",
                        "Regular medical monitoring essential",
                        "Combination of diet, exercise, and potential medication"
                    ],
                    "Obesity_Type_III": [
                        "Urgent medical consultation required",
                        "Comprehensive multidisciplinary approach needed",
                        "May require surgical intervention evaluation",
                        "Close medical supervision essential"
                    ]
                }
                
                rec_list = recommendations.get(prediction_label, [
                    "Maintain a balanced diet",
                    "Stay physically active",
                    "Consult with a healthcare professional for personalized advice"
                ])
                
                for i, rec in enumerate(rec_list, 1):
                    st.info(f"{i}. {rec}")
                    
                # Add a button to reset and make new prediction
                if st.button("🔄 Make Another Prediction", use_container_width=True):
                    st.session_state.prediction_made = False
                    st.rerun()
            
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.code(f"Error details:\n{str(e)}", language="python")
                
                # Debug information
                with st.expander("🔧 Technical Details"):
                    st.write("**Error type:**", type(e).__name__)
                    if 'selected_features' in locals():
                        st.write("**Selected features:**", selected_features)
                    if 'full_df' in locals():
                        st.write("**DataFrame columns:**", list(full_df.columns))
                        st.write("**DataFrame shape:**", full_df.shape)
        
        elif st.session_state.prediction_made:
            # Show previous prediction
            st.markdown('<h2 class="sub-header">📊 Previous Prediction</h2>', unsafe_allow_html=True)
            
            prediction_label = st.session_state.prediction_result
            
            # Display previous prediction
            prediction_colors = {
                "Insufficient_Weight": "#FF6B6B",
                "Normal_Weight": "#4ECDC4",
                "Overweight_Level_I": "#FFD166",
                "Overweight_Level_II": "#FF9A76",
                "Obesity_Type_I": "#FF6B6B",
                "Obesity_Type_II": "#FF4757",
                "Obesity_Type_III": "#FF3838"
            }
            
            color = prediction_colors.get(prediction_label, "#1E3A8A")
            
            st.markdown(f"""
            <div class="prediction-box" style="border-left-color: {color};">
                <h3 style="color: {color};">🎯 Previous Prediction</h3>
                <h1 style="color: {color}; text-align: center; margin: 1rem 0;">{prediction_label}</h1>
                <p style="text-align: center;">
                    Based on your previous input data.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 Make New Prediction", use_container_width=True):
                st.session_state.prediction_made = False
                st.rerun()
        
        else:
            # Show placeholder before prediction
            st.markdown('<h2 class="sub-header">📊 Prediction Results</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; background-color: #F8F9FA; border-radius: 10px;">
                <h3 style="color: #6B7280;">👈 Enter your information</h3>
                <p>Fill out the form on the left and click "Predict Obesity Level" to see your results here.</p>
                <p><small>Model loaded: ✅</small></p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem; margin-top: 2rem;">
    <p>⚠️ <strong>Disclaimer:</strong> This tool provides predictions based on machine learning models and should not replace professional medical advice. Always consult with healthcare professionals for medical decisions.</p>
    <p>🔒 Your data is processed locally and not stored.</p>
    <p>📦 <strong>Model Source:</strong> {model_source}</p>
</div>
""".format(
    model_source=st.session_state.get('model_source', 'GitHub: yaswanth-nayana/Obesity')
), unsafe_allow_html=True)
