import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt

# Load data
print("Loading data...")
data = pd.read_csv(Path(__file__).parent / "heart.csv")
print(f"Data has {len(data)} rows")

# Prepare data
cats = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
data = pd.get_dummies(data, columns=cats)

num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "K-Nearest Neighbors Classifier": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(probability=True),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier()
}

best_score = 0
best_model = None
best_name = ""

# For storing metrics for visualization
model_names = []
accuracies = []
precisions = []
recalls = []
f1_scores = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1-Score: {f1:.2f}")

    overall_score = (acc + prec + rec + f1) / 4
    if overall_score > best_score:
        best_score = overall_score
        best_model = model
        best_name = name

    model_names.append(name)
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

print(f"\n✅ Best model is {best_name} with average score {best_score:.2f}")

# Save best model and scaler
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("Models saved")

# Create separate graphs for each metric
metrics_data = {
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1 Score': f1_scores
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for metric_name, metric_values in metrics_data.items():
    plt.figure(figsize=(10, 6))
    
    # Create bar chart for current metric
    bars = plt.bar(model_names, metric_values, color=colors, alpha=0.8)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{metric_values[i]:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Customize the plot
    plt.title(f'{metric_name} Comparison Across Models', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Models', fontsize=12, fontweight='bold')
    plt.ylabel(f'{metric_name} Score', fontsize=12, fontweight='bold')
    plt.ylim(0, max(metric_values) + 0.1)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Highlight the best performing model
    best_idx = metric_values.index(max(metric_values))
    bars[best_idx].set_color('#ff4444')
    bars[best_idx].set_alpha(1.0)
    
    plt.tight_layout()
    plt.show()

# Function to load model and make a prediction
def predict_heart_disease(patient_data):
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    import pandas as pd
    df = pd.DataFrame([patient_data])
    df = pd.get_dummies(df)

    # Add missing columns with 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cols_to_scale = [col for col in num_cols if col in df.columns]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    prediction = model.predict(df)[0]
    return prediction

# Example usage
if __name__ == "__main__":
    sample_patient = {
        'age': 63,
        'sex': 1,
        'cp': 3,
        'trestbps': 145,
        'chol': 233,
        'fbs': 1,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 2.3,
        'slope': 0,
        'ca': 0,
        'thal': 1
    }

    result = predict_heart_disease(sample_patient)
    if result == 1:
        print("Prediction: Heart Disease Risk")
    else:
        print("Prediction: Low Heart Disease Risk")
