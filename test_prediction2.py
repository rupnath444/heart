import pickle
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

def get_input(prompt, valid_type=float, valid_values=None):
    while True:
        try:
            user_input = valid_type(input(prompt))
            if valid_values is not None and user_input not in valid_values:
                print(f"Please enter one of the following valid options: {valid_values}")
                continue
            return user_input
        except ValueError:
            print(f"Invalid input. Please enter a valid {valid_type.__name__}.")

def get_user_data():
    print("🏥 Heart Disease Risk Assessment")
    print("="*40)
    print("Please enter your health information:")
    print("-"*40)

    print("\n📋 Personal Information:")
    print("Age: Your current age in years")
    age = get_input("Age: ", float)

    print("\n👤 Sex:")
    print("ℹ️ Men often have higher heart disease risk at younger ages; women's risk rises after menopause.")
    print("0 = Female")
    print("1 = Male")
    sex = get_input("Enter choice (0 or 1): ", int, [0,1])

    print("\n💔 Chest Pain Type:")
    print("ℹ️ Chest pain caused by reduced blood flow to the heart.")
    print("0 = Typical Angina: crushing chest pain during exercise; classic sign of blocked arteries.")
    print("1 = Atypical Angina: unusual chest pain, could indicate heart problems.")
    print("2 = Non-Anginal Pain: chest pain unrelated to heart, e.g. muscle strain.")
    print("3 = Asymptomatic: no chest pain but possible silent heart disease.")
    cp = get_input("Chest Pain Type (0-3): ", int, [0,1,2,3])

    print("\n🩺 Resting Blood Pressure (mmHg):")
    print("ℹ️ Pressure when heart is at rest. Normal <120; high >140.")
    trestbps = get_input("Resting Blood Pressure (e.g. 120): ", float)

    print("\n🧪 Cholesterol Level (mg/dl):")
    print("ℹ️ Fatty substance that can clog arteries. Normal <200; borderline 200-239; high >240.")
    chol = get_input("Cholesterol Level (e.g. 200): ", float)

    print("\n🍬 Fasting Blood Sugar:")
    print("ℹ️ High sugar damages vessels; diabetics have 2-4x heart disease risk.")
    print("0 = No (≤120 mg/dl)")
    print("1 = Yes (>120 mg/dl)")
    fbs = get_input("Fasting Blood Sugar > 120 mg/dl? (0 or 1): ", int, [0,1])

    print("\n📈 Resting Electrocardiographic Results:")
    print("0 = Normal rhythm")
    print("1 = ST-T Wave Abnormality (minor changes)")
    print("2 = Left Ventricular Hypertrophy (enlarged heart chamber)")
    restecg = get_input("Resting ECG results (0-2): ", int, [0,1,2])

    print("\n💓 Maximum Heart Rate Achieved:")
    print("Highest heart rate during exercise; lower values can indicate problems.")
    thalach = get_input("Max Heart Rate (e.g. 150): ", float)

    print("\n🏃 Exercise Induced Angina:")
    print("Chest pain only during physical activity.")
    print("0 = No")
    print("1 = Yes")
    exang = get_input("Angina during exercise? (0 or 1): ", int, [0,1])

    print("\n📊 ST Depression Induced by Exercise (Oldpeak):")
    print("Shows heart strain during exercise test, typical range 0.0 - 4.0")
    oldpeak = get_input("ST Depression value (e.g. 1.5): ", float)

    print("\n📉 Slope of the Peak Exercise ST Segment:")
    print("0 = Upsloping (best)")
    print("1 = Flat")
    print("2 = Downsloping (concerning)")
    slope = get_input("Slope type (0-2): ", int, [0,1,2])

    print("\n🩻 Number of Major Vessels Colored by Fluoroscopy:")
    print("0 = None, 3 = 3 or more vessels blocked")
    ca = get_input("Blocked vessels (0-3): ", int, [0,1,2,3])

    print("\n🩸 Thalassemia Status:")
    print("1 = Normal")
    print("2 = Fixed Defect (permanent)")
    print("3 = Reversible Defect (temporary)")
    thal = get_input("Thalassemia status (1-3): ", int, [1,2,3])

    return {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

def load_model():
    if not os.path.exists('models'):
        print("Model folder not found. Please train the model first.")
        return None, None, None
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

def preprocess_input(patient_data, feature_names, scaler):
    df = pd.DataFrame([patient_data])
    df = pd.get_dummies(df)
    # Add missing columns with 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    # Scale numeric columns
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df

def predict_risk(processed_df, model):
    prediction = model.predict(processed_df)[0]
    try:
        proba = model.predict_proba(processed_df)[0][1]
    except:
        proba = None
    if proba is not None:
        if proba >= 0.7:
            risk = "High Risk"
        elif proba >= 0.4:
            risk = "Moderate Risk"
        else:
            risk = "Low Risk"
        probability_pct = proba * 100
    else:
        risk = "High Risk" if prediction == 1 else "Low Risk"
        probability_pct = None
    return prediction, risk, probability_pct

def main():
    print("🫀 Heart Disease Risk Predictor")
    model, scaler, feature_names = load_model()
    if model is None:
        return

    while True:
        patient_data = get_user_data()
        processed = preprocess_input(patient_data, feature_names, scaler)
        prediction, risk, probability = predict_risk(processed, model)

        print("\n" + "="*50)
        print("🎯 YOUR HEART DISEASE RISK ASSESSMENT")
        print("="*50)
        print(f"Prediction: {'Heart Disease Risk' if prediction == 1 else 'Low Heart Disease Risk'}")
        print(f"Risk Level: {risk}")
        if probability is not None:
            print(f"Probability of Disease: {probability:.1f}%")

        again = input("Would you like to check another patient? (y/n): ").lower()
        if again not in ['y', 'yes']:
            print("Thank you for using the Heart Disease Predictor. Stay healthy!")
            break

if __name__ == "__main__":
    main()
