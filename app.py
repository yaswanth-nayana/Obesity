import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os
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
    .feature-importance {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stButton button {
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_data' not in st.session_state:
    st.session_state.model_data = None

# App title and description
st.markdown('<h1 class="main-header">⚖️ Obesity Level Prediction System</h1>', unsafe_allow_html=True)

st.markdown("""
This application uses machine learning to predict obesity levels based on lifestyle and dietary habits. 
Enter your information below to get a prediction and personalized insights.
""")

# Sidebar for model loading and information
with st.sidebar:
    st.header("🔧 Model Configuration")
    
    # File selection for loading model
    st.subheader("Load Model Files")
    
    # Check if files exist
    pkg_exists = os.path.exists('model_package.pkl')
    model_exists = os.path.exists('best_model.pkl')
    
    if not pkg_exists:
        st.error("❌ model_package.pkl not found in current directory!")
    if not model_exists:
        st.warning("⚠️ best_model.pkl not found in current directory!")
    
    # Load model button
    if st.button("🔄 Load Models", type="primary", use_container_width=True):
        try:
            # Load model package first
            if pkg_exists:
                with open('model_package.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                
                # Verify it's a valid pickle file
                if isinstance(model_data, dict) and 'model' in model_data:
                    st.session_state.model_data = model_data
                    st.session_state.model_loaded = True
                    st.success("✅ Model package loaded successfully!")
                    
                    # Also try to load standalone model
                    if model_exists:
                        try:
                            with open('best_model.pkl', 'rb') as f:
                                standalone_model = pickle.load(f)
                            st.info(f"✓ Both pickle files loaded successfully")
                        except:
                            st.warning("Standalone model file may be corrupted")
                else:
                    st.error("Invalid model package format!")
            else:
                st.error("model_package.pkl file not found!")
                
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            st.info("""
            **Possible causes:**
            1. File is corrupted or empty
            2. File contains text instead of binary data
            3. Wrong file format
            """)
    
    st.divider()
    
    # Model information section
    if st.session_state.model_loaded:
        model_data = st.session_state.model_data
        st.header("📊 Model Information")
        
        st.info(f"**Model Type:** {type(model_data['model']).__name__}")
        st.info(f"**Features Used:** {len(model_data['top_features'])}")
        
        with st.expander("View Selected Features"):
            for i, feature in enumerate(model_data['top_features'], 1):
                st.write(f"{i}. {feature}")
        
        st.info(f"**Scaler:** {type(model_data['scaler']).__name__}")
        st.info(f"**Label Encoders:** {len(model_data.get('label_encoders', {}))}")
    
    st.divider()
    
    # Troubleshooting section
    st.header("🛠️ Troubleshooting")
    st.markdown("""
    If models fail to load:
    1. **Run your training script again** to regenerate pickle files
    2. **Check file sizes:**
       - model_package.pkl should be > 1KB
       - best_model.pkl should be > 1KB
    3. **Try loading manually:**
    ```python
    import pickle
    with open('model_package.pkl', 'rb') as f:
        data = pickle.load(f)
    print(type(data))
    ```
    """)

# Main content area
if not st.session_state.model_loaded:
    # Show warning if model not loaded
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.warning("""
        ## ⚠️ Models Not Loaded
        
        Please click **"Load Models"** in the sidebar to load your trained models.
        
        **Required files in current directory:**
        1. `model_package.pkl` - Main model package (required)
        2. `best_model.pkl` - Standalone model (optional)
        """)
        
        # Show current directory files
        with st.expander("Check Current Directory Files"):
            try:
                files = os.listdir('.')
                pkl_files = [f for f in files if f.endswith('.pkl')]
                if pkl_files:
                    st.write("Found pickle files:", pkl_files)
                    for file in pkl_files:
                        size = os.path.getsize(file)
                        st.write(f"- {file}: {size} bytes")
                else:
                    st.write("No .pkl files found in current directory")
            except:
                st.write("Could not read directory")
else:
    # Model is loaded - show the prediction interface
    model_data = st.session_state.model_data
    model = model_data['model']
    scaler = model_data['scaler']
    top_features = model_data['top_features']
    label_encoders = model_data.get('label_encoders', {})
    
    # Initialize prediction session state
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    
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
                gender = st.selectbox(
                    "Gender",
                    options=["Male", "Female"],
                    help="Select your gender"
                )
            
            with col_b:
                age = st.number_input(
                    "Age",
                    min_value=1,
                    max_value=120,
                    value=30,
                    help="Enter your age in years"
                )
            
            # Physical measurements
            col_c, col_d = st.columns(2)
            with col_c:
                height = st.number_input(
                    "Height (meters)",
                    min_value=0.5,
                    max_value=2.5,
                    value=1.7,
                    step=0.01,
                    format="%.2f",
                    help="Enter your height in meters"
                )
            
            with col_d:
                weight = st.number_input(
                    "Weight (kg)",
                    min_value=20.0,
                    max_value=300.0,
                    value=70.0,
                    step=0.1,
                    format="%.1f",
                    help="Enter your weight in kilograms"
                )
            
            # Calculate BMI in real-time
            bmi = weight / (height ** 2)
            st.metric("Your BMI", f"{bmi:.1f}", 
                     delta="Healthy" if 18.5 <= bmi <= 24.9 else 
                            "Underweight" if bmi < 18.5 else "Overweight")
            
            st.divider()
            st.subheader("Family History & Habits")
            
            # Family history and habits
            family_history = st.selectbox(
                "Family History with Overweight",
                options=["yes", "no"],
                help="Does your family have a history of overweight?"
            )
            
            favc = st.selectbox(
                "Frequent consumption of high caloric food",
                options=["yes", "no"],
                help="Do you frequently eat high calorie food?"
            )
            
            st.divider()
            st.subheader("Dietary Habits")
            
            # Dietary habits in columns
            col_e, col_f = st.columns(2)
            with col_e:
                fcvc = st.slider(
                    "Frequency of vegetable consumption",
                    min_value=1.0,
                    max_value=3.0,
                    value=2.0,
                    step=0.1,
                    help="How often do you eat vegetables? (1=Never, 3=Always)"
                )
            
            with col_f:
                ncp = st.slider(
                    "Number of main meals",
                    min_value=1.0,
                    max_value=4.0,
                    value=3.0,
                    step=0.1,
                    help="How many main meals do you have daily?"
                )
            
            # More dietary habits
            col_g, col_h = st.columns(2)
            with col_g:
                caec = st.selectbox(
                    "Consumption of food between meals",
                    options=["no", "Sometimes", "Frequently", "Always"],
                    help="Do you eat between meals?"
                )
            
            with col_h:
                calc = st.selectbox(
                    "Consumption of alcohol",
                    options=["no", "Sometimes", "Frequently", "Always"],
                    help="How often do you consume alcohol?"
                )
            
            st.divider()
            st.subheader("Physical Activity & Lifestyle")
            
            # Physical activity
            faf = st.slider(
                "Physical activity frequency",
                min_value=0.0,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="How often are you physically active? (0=Never, 3=Very Often)"
            )
            
            # Transportation
            mtrans = st.selectbox(
                "Transportation used",
                options=["Public_Transportation", "Automobile", "Walking", "Bike", "Motorbike"],
                help="What is your primary mode of transportation?"
            )
            
            # Additional lifestyle factors
            col_i, col_j = st.columns(2)
            with col_i:
                smoke = st.selectbox(
                    "Smoking habit",
                    options=["yes", "no"],
                    help="Do you smoke?"
                )
            
            with col_j:
                scc = st.selectbox(
                    "Calories consumption monitoring",
                    options=["yes", "no"],
                    help="Do you monitor your calorie intake?"
                )
            
            # Water consumption and technology use
            col_k, col_l = st.columns(2)
            with col_k:
                ch2o = st.slider(
                    "Daily water consumption (liters)",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                    help="How much water do you drink daily?"
                )
            
            with col_l:
                tue = st.slider(
                    "Time using technology devices",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.5,
                    step=0.1,
                    help="Daily hours using electronic devices"
                )
            
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
            <div class="prediction-box">
                <h3>📈 Your BMI: {bmi:.1f}</h3>
                <p><strong>BMI Category:</strong> {bmi_category}</p>
                <p><strong>Height:</strong> {height}m | <strong>Weight:</strong> {weight}kg</p>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                # Prepare data for prediction - FIXED SECTION
                prediction_data = {}
                
                # Encode categorical features
                for feature, value in user_data.items():
                    if feature in label_encoders:
                        try:
                            prediction_data[feature] = label_encoders[feature].transform([value])[0]
                        except ValueError:
                            # Handle unknown labels
                            all_classes = list(label_encoders[feature].classes_)
                            if all_classes:
                                prediction_data[feature] = label_encoders[feature].transform([all_classes[0]])[0]
                            else:
                                prediction_data[feature] = 0
                    else:
                        prediction_data[feature] = value
                
                # Define ALL original features (16 features) that the scaler expects
                all_original_features = [
                    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
                    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
                    'CALC', 'MTRANS'
                ]
                
                # Create DataFrame with ALL 16 features
                full_df = pd.DataFrame(columns=all_original_features)
                
                # Fill the DataFrame with our data (all features should be present)
                for feature in all_original_features:
                    if feature in prediction_data:
                        full_df[feature] = [prediction_data[feature]]
                    else:
                        # This shouldn't happen since we collected all features from user
                        # But just in case, fill with default values
                        if feature in ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']:
                            full_df[feature] = [0.0]  # Default for numerical
                        else:
                            full_df[feature] = [0]    # Default for categorical
                
                # Scale ALL 16 features (as the scaler expects)
                scaled_data = scaler.transform(full_df)
                
                # Now extract only the top 8 features for the model
                if top_features:
                    # Get indices of top features in the scaled data
                    top_feature_indices = []
                    for feature in top_features:
                        if feature in all_original_features:
                            idx = all_original_features.index(feature)
                            top_feature_indices.append(idx)
                    
                    # Extract only the top features from scaled data
                    if top_feature_indices:
                        scaled_top_features = scaled_data[:, top_feature_indices]
                    else:
                        scaled_top_features = scaled_data
                else:
                    scaled_top_features = scaled_data
                
                # Make prediction using only top features
                prediction = model.predict(scaled_top_features)[0]
                
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
                
                # Show feature importance insights
                st.markdown('<h3 class="sub-header">💡 Key Influencing Factors</h3>', unsafe_allow_html=True)
                
                # Create a simple visualization of important features
                important_features = {
                    'Weight': weight,
                    'Height': height,
                    'Age': age,
                    'Physical Activity': faf,
                    'Vegetable Consumption': fcvc,
                    'Water Intake': ch2o
                }
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(important_features.keys()),
                        y=list(important_features.values()),
                        marker_color=['#1E3A8A', '#3B82F6', '#60A5FA', '#93C5FD', '#BFDBFE', '#DBEAFE'],
                        text=[f"{v:.1f}" for v in important_features.values()],
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title="Your Current Metrics",
                    xaxis_title="Features",
                    yaxis_title="Value",
                    height=400,
                    showlegend=False,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
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
            
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.code(f"Error details: {str(e)}", language="python")
        
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
</div>
""", unsafe_allow_html=True)

# Debug information (hidden by default)
with st.expander("Debug Information"):
    st.write("Session state:", st.session_state)
    if st.session_state.model_loaded:
        st.write("Model type:", type(model).__name__)
        st.write("Top features:", top_features)
        st.write("Scaler type:", type(scaler).__name__)
        st.write("Number of features scaler expects:", scaler.n_features_in_)