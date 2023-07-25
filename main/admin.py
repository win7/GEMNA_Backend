from django.contrib import admin
from main.models import *

# Register your models here.

@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
	"""docstring for ExperimentAdmin"""
	list_display = ["method", "data_variation", "dimension", "runtime", "raw_data"]