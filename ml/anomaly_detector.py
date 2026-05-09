import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

def generate_sample_data():
    np.random.seed(42)
    normal = np.random.normal(loc=50, scale=5, size=(200, 3))
    faulty = np.random.normal(loc=75, scale=10, size=(50, 3))

    X = np.vstack([normal, faulty])
    y = np.array([0] * 200 + [1] * 50)  # 0=normal, 1=faulty
    return X, y

def train_model():
    X, y = generate_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test)))

    with open("ml/model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to ml/model.pkl")

def predict(sensor_readings: list):
    with open("ml/model.pkl", "rb") as f:
        model = pickle.load(f)
    prediction = model.predict([sensor_readings])
    return "faulty" if prediction[0] == 1 else "normal"

if __name__ == "__main__":
    train_model()
    print("\nTest prediction:", predict([72, 80, 68]))