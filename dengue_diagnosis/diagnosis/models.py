from django.db import models

# Create your models here.
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example data
symptoms = [
    [1, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 1, 1, 1],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    [1, 0, 0, 1],
]
labels = [1, 1, 0, 1, 0, 0, 0]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(symptoms, labels, test_size=0.2)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
print(f'Model Accuracy: {accuracy_score(y_test, y_pred) * 100}%')

# Saving the model
joblib.dump(model, 'dengue_model.pkl')  # Saving the model to the current directory



