from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns

from main.api_views import *

urlpatterns = [
    path('experiments/', ExperimentList.as_view(), name="experiments"),
    path('experiments/<slug:pk>/', ExperimentDetail.as_view(), name="experiments-detail"),
    path('experiments-consult/', ExperimentConsult.as_view(), name="experiments-consult"),
    path('experiments-finetune/', ExperimentFinetune.as_view(), name="experiments-finetune"),
]
urlpatterns = format_suffix_patterns(urlpatterns)