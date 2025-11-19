from django.urls import path
from . import views

# This list defines the URL patterns for the app.
urlpatterns = [
    # Maps the root URL of the app to the 'index' view in views.py
    # When a user visits the base URL (e.g., /), Django will call the index function.
    path('', views.index, name='index'),
]

