from django.contrib import admin
from main.models import *

# Register your models here.

@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
	"""docstring for ExperimentAdmin"""
	list_display = ["id", "created_at", "method", "data_variation", "dimension",
		 			"control", "transformation", "range", "alpha", "runtime", "raw_data", "detail"]