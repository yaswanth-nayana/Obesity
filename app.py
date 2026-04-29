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


def get_bmi_target_class(bmi):
    """Return the target class that matches the BMI ranges used in the dataset."""
    if bmi < 18.5:
        return "Insufficient_Weight", "BMI < 18.5 - underweight individuals"
    if bmi < 25.0:
        return "Normal_Weight", "BMI 18.5-24.9 - healthy weight range"
    if bmi < 27.5:
        return "Overweight_Level_I", "BMI 25.0-27.4 - mild overweight"
    if bmi < 30.0:
        return "Overweight_Level_II", "BMI 27.5-29.9 - moderate overweight"
    if bmi < 35.0:
        return "Obesity_Type_I", "BMI 30.0-34.9 - Class I obesity"
    if bmi < 40.0:
        return "Obesity_Type_II", "BMI 35.0-39.9 - Class II obesity (severe)"
    return "Obesity_Type_III", "BMI >= 40.0 - Class III obesity"


def get_lifestyle_case(favc, fcvc, ncp, caec, calc, scc, faf, ch2o, tue, mtrans):
    """Summarize dietary and physical-activity indicators used with the BMI class."""
    diet_flags = []
    activity_flags = []

    if favc == "yes":
        diet_flags.append("frequent high-calorie food")
    if fcvc < 2.0:
        diet_flags.append("low vegetable intake")
    if ncp > 3.0:
        diet_flags.append("more than 3 main meals")
    if caec in ["Frequently", "Always"]:
        diet_flags.append("frequent eating between meals")
    if calc in ["Frequently", "Always"]:
        diet_flags.append("frequent alcohol consumption")
    if scc == "no":
        diet_flags.append("no calorie monitoring")
    if ch2o < 1.5:
        diet_flags.append("low water intake")

    if faf < 1.0:
        activity_flags.append("low physical activity")
    if tue > 1.5:
        activity_flags.append("high technology/sedentary time")
    if mtrans in ["Automobile", "Motorbike"]:
        activity_flags.append("motorized transportation")

    diet_score = len(diet_flags)
    activity_score = len(activity_flags)
    total_score = diet_score + activity_score

    if total_score >= 5:
        case = "High lifestyle risk"
    elif total_score >= 3:
        case = "Moderate lifestyle risk"
    elif total_score >= 1:
        case = "Low lifestyle risk"
    else:
        case = "Healthy lifestyle indicators"

    return {
        "case": case,
        "diet_score": diet_score,
        "activity_score": activity_score,
        "diet_flags": diet_flags or ["diet indicators are favorable"],
        "activity_flags": activity_flags or ["activity indicators are favorable"]
    }

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
            bmi_target, bmi_description = get_bmi_target_class(bmi)
            st.markdown(f"""
            <div class="metric-card">
                <h4>Your BMI</h4>
                <h2>{bmi:.1f}</h2>
                <p><strong>Target Class:</strong> {bmi_target}</p>
                <p><strong>Range:</strong> {bmi_description}</p>
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
                # Prepare raw input first, then encode with saved encoders when available.
                raw_input_data = {
                    'Gender': gender,
                    'Age': float(age),
                    'Height': float(height),
                    'Weight': float(weight),
                    'family_history_with_overweight': family_history,
                    'FAVC': favc,
                    'FCVC': float(fcvc),
                    'NCP': float(ncp),
                    'CAEC': caec,
                    'SMOKE': smoke,
                    'CH2O': float(ch2o),
                    'SCC': scc,
                    'FAF': float(faf),
                    'TUE': float(tue),
                    'CALC': calc,
                    'MTRANS': mtrans
                }

                manual_maps = {
                    "Gender": {"Male": 0, "Female": 1},
                    "family_history_with_overweight": {"yes": 1, "no": 0},
                    "FAVC": {"yes": 1, "no": 0},
                    "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
                    "SMOKE": {"yes": 1, "no": 0},
                    "SCC": {"yes": 1, "no": 0},
                    "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
                    "MTRANS": {
                        "Public_Transportation": 0,
                        "Automobile": 1,
                        "Walking": 2,
                        "Bike": 3,
                        "Motorbike": 4
                    }
                }

                input_data = {}
                for feature, value in raw_input_data.items():
                    if feature in label_encoders:
                        try:
                            input_data[feature] = float(label_encoders[feature].transform([value])[0])
                            continue
                        except Exception:
                            pass

                    if feature in manual_maps:
                        input_data[feature] = float(manual_maps[feature][value])
                    else:
                        input_data[feature] = float(value)

                # Resolve model feature space separately from scaler feature space.
                raw_model_features = []
                if hasattr(model, 'feature_name_') and len(getattr(model, 'feature_name_', [])) > 0:
                    raw_model_features = list(model.feature_name_)
                elif hasattr(model, 'feature_names_in_'):
                    raw_model_features = list(model.feature_names_in_)

                # LightGBM can store generic names like Column_0..Column_N.
                uses_generic_lgbm_names = (
                    len(raw_model_features) > 0 and
                    all(str(name).startswith("Column_") for name in raw_model_features)
                )

                if top_features and len(top_features) > 0 and (
                    uses_generic_lgbm_names or len(raw_model_features) == 0
                ):
                    model_features = list(top_features)
                elif len(raw_model_features) > 0:
                    model_features = raw_model_features
                else:
                    model_features = list(input_data.keys())

                if scaler and hasattr(scaler, 'feature_names_in_'):
                    scaler_features = list(scaler.feature_names_in_)
                else:
                    scaler_features = list(model_features)

                st.info(f"**Input features prepared:** {len(input_data)}")
                st.info(f"**Scaler expects features:** {len(scaler_features)}")
                st.info(f"**Model expects features:** {len(model_features)}")

                missing_scaler_features = [f for f in scaler_features if f not in input_data]
                missing_model_features = [f for f in model_features if f not in input_data]
                if missing_scaler_features:
                    st.warning(f"Missing {len(missing_scaler_features)} scaler features; using 0 for those fields.")
                if missing_model_features:
                    st.warning(f"Missing {len(missing_model_features)} model features; using 0 for those fields.")

                # Step 1: Build scaler input (16 features for your current artifacts).
                scaler_df = pd.DataFrame(
                    {feature: [input_data.get(feature, 0.0)] for feature in scaler_features},
                    columns=scaler_features
                )

                if scaler:
                    try:
                        scaled_values = scaler.transform(scaler_df)
                        scaled_df = pd.DataFrame(scaled_values, columns=scaler_features)
                        st.success("Data scaled successfully with scaler")
                    except Exception as e:
                        st.error(f"Scaler transformation failed: {str(e)}")
                        scaled_df = scaler_df.copy()
                else:
                    scaled_df = scaler_df.copy()

                # Step 2: Select the exact model feature set (8 features for your current model).
                model_df = pd.DataFrame(
                    {feature: [input_data.get(feature, 0.0)] for feature in model_features},
                    columns=model_features
                )
                for feature in model_features:
                    if feature in scaled_df.columns:
                        model_df[feature] = scaled_df[feature].values

                model_input = model_df.values
                # Make prediction
                with st.spinner("Making prediction..."):
                    prediction = model.predict(model_input)[0]
                if hasattr(prediction, "item"):
                    prediction = prediction.item()
                
                # Decode prediction based on your value_counts
                obesity_labels = [
                    "Obesity_Type_III",       # Class 0
                    "Obesity_Type_II",        # Class 1
                    "Normal_Weight",          # Class 2
                    "Obesity_Type_I",         # Class 3
                    "Insufficient_Weight",    # Class 4
                    "Overweight_Level_II",    # Class 5
                    "Overweight_Level_I"      # Class 6
                ]
                
                result = ""
                if isinstance(prediction, str):
                    result = prediction
                elif 'NObeyesdad' in label_encoders:
                    try:
                        result = label_encoders['NObeyesdad'].inverse_transform([prediction])[0]
                    except:
                        pred_idx = int(prediction)
                        if pred_idx < len(obesity_labels):
                            result = obesity_labels[pred_idx]
                        else:
                            result = f"Class {pred_idx}"
                else:
                    pred_idx = int(prediction)
                    if pred_idx < len(obesity_labels):
                        result = obesity_labels[pred_idx]
                    else:
                        result = f"Class {pred_idx}"

                model_result = result
                bmi_result, bmi_description = get_bmi_target_class(bmi)
                lifestyle_case = get_lifestyle_case(
                    favc, fcvc, ncp, caec, calc, scc, faf, ch2o, tue, mtrans
                )
                result = bmi_result
                
                # Store result
                st.session_state.prediction_result = result
                st.session_state.prediction_details = {
                    "bmi": bmi,
                    "bmi_description": bmi_description,
                    "model_result": model_result,
                    "raw_prediction": prediction,
                    "lifestyle_case": lifestyle_case
                }
                st.session_state.prediction_made = True
                
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
                        {bmi_description}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.info(f"**Model output using dietary/activity inputs:** {model_result} (raw class: {prediction})")
                if model_result != result:
                    st.warning(
                        "The ML model output differs from the BMI target class. "
                        "The main class above follows the BMI ranges from your target-variable table."
                    )

                st.markdown('<h3>Dietary and Physical Activity Case</h3>', unsafe_allow_html=True)
                case_col1, case_col2 = st.columns(2)
                with case_col1:
                    st.write(f"**Dietary score:** {lifestyle_case['diet_score']}")
                    for flag in lifestyle_case["diet_flags"]:
                        st.write(f"- {flag}")
                with case_col2:
                    st.write(f"**Physical activity score:** {lifestyle_case['activity_score']}")
                    for flag in lifestyle_case["activity_flags"]:
                        st.write(f"- {flag}")
                st.info(f"**Combined case:** {lifestyle_case['case']}")
                
                # Metrics
                st.markdown('<h3>📈 Key Health Factors</h3>', unsafe_allow_html=True)
                
                metrics = [
                    ("Weight", weight / 150),
                    ("Age", age / 80),
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
                with st.expander("🔧 Debug Details"):
                    st.write("**Error:**", str(e))
                    st.write("**Model type:**", type(model).__name__)
                    if scaler:
                        st.write("**Scaler type:**", type(scaler).__name__)
                        if hasattr(scaler, 'feature_names_in_'):
                            st.write("**Scaler features:**", list(scaler.feature_names_in_))
                    st.write("**Input data keys:**", list(input_data.keys()))
                    st.write("**Model features:**", model_features if 'model_features' in locals() else "Not resolved")
        
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
                    Based on your previous input data.
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
