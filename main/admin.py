from django.contrib import admin
from main.models import *

# Register your models here.

@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
	"""docstring for ExperimentAdmin"""
	list_display = ["id", "created_at", "method", "data_variation", "dimension",
		 			"controls", "transformation", "threshold_corr", "threshold_log2", "alpha", "runtime", "raw_data", "detail"]