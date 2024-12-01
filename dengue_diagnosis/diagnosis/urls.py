# diagnosis/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_dengue, name='predict_dengue'),
]

