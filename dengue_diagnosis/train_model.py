# train_model.py (Create this file to train your model)

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example data: each row represents [fever, pain, rash, other_symptoms]
# 1 = Present, 0 = Absent
symptoms = [
    [1, 1, 0, 1],  # Fever, Pain, Rash, Other symptoms (1 = Present, 0 = Absent)
    [1, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 1, 1, 1],
    [0, 1, 1, 0]
]

# Labels: 1 = Dengue, 0 = No Dengue
labels = [1, 1, 0, 1, 0]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(symptoms, labels, test_size=0.2)

# Train the Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test the model and print accuracy
y_pred = model.predict(X_test)
print(f'Model Accuracy: {accuracy_score(y_test, y_pred) * 100}%')

# Save the trained model
joblib.dump(model, 'dengue_model.pkl')  # Save model to a file
