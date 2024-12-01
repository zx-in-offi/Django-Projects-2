from django.shortcuts import render

# Create your views here.
# diagnosis/views.py
# diagnosis/views.py
# diagnosis/views.py
import joblib
import os
from django.shortcuts import render
from django.http import JsonResponse

# Get the absolute path to the model file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'dengue_model.pkl')

# Load the trained model
model = joblib.load(model_path)

# diagnosis/views.py
def predict_dengue(request):
    if request.method == 'POST':
        # Use .get() to avoid errors if key is missing
        fever = int(request.POST.get('fever', 0))  # Default to 0 if 'fever' is missing
        pain = int(request.POST.get('pain', 0))  # Default to 0 if 'pain' is missing
        rash = int(request.POST.get('rash', 0))  # Default to 0 if 'rash' is missing
        other_symptoms = int(request.POST.get('other_symptoms', 0))  # Default to 0 if 'other_symptoms' is missing

        # Create a list of symptoms to predict
        symptoms = [[fever, pain, rash, other_symptoms]]

        # Predict dengue
        prediction = model.predict(symptoms)

        # Return the result
        result = 'Dengue Detected' if prediction == 1 else 'No Dengue'
        return JsonResponse({'prediction': result})

    return render(request, 'diagnosis/predict_dengue.html')



