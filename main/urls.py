from django.urls import path
from main.api_views import *

urlpatterns = [
    path('experiments/', ExperimentList.as_view(), name="experiment"),
    # path('snippets/<int:pk>/', views.snippet_detail),
]