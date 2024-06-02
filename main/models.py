from django.db import models

import uuid

# Create your models here.

method_choice = (
    ("vgae-line", "LVGAE"),
	("vgae-base", "VGAE"),
	("argva-base", "ARGVA"),
	("dgi-tran", "DGI"), 
)

option_choice = (
	("none", "none"), 
	("str", "str"),
	("dyn", "dyn")
)

status_choice = (
	("s1", "start"), 
	("s2", "processing"),
	("s2", "end")
)

class Experiment(models.Model):
	"""docstring for Experiment"""
	id = models.UUIDField("Id", primary_key=True, default=uuid.uuid4, editable=False)
	created_at = models.DateTimeField("Created", auto_now_add=True)
	updated_at = models.DateTimeField("Updated", auto_now=True)
	method = models.CharField("Method", choices=method_choice, default=method_choice[0][0], max_length=15)
	data_variation = models.CharField("Dataset variation", choices=option_choice, default=option_choice[0][0], max_length=4)
	dimension = models.IntegerField("Dimension", default=3)
	raw_data = models.FileField("Raw data", upload_to='experiments/raw_data/', blank=False, null=False)
	controls = models.CharField("Controls", max_length=64)
	range = models.IntegerField("Range", default=0)
	alpha = models.FloatField("Alpha", default=0.05)
	threshold_corr = models.FloatField("Threshold (corr)", default=0.5)
	threshold_log2 = models.IntegerField("Threshold (log2)", default=0)
	silhouette = models.FloatField("Silhouette score", default=0.0)
	runtime = models.FloatField("Runtime", default=0.0)
	email = models.EmailField("Email", blank=False, null=False)
	transformation = models.BooleanField("Transformation", default=False)
	detail = models.TextField("Details", max_length=100, blank=True, null=True)
	
	# profile_id = models.OneToOneField(Profile, on_delete=models.CASCADE, verbose_name="Perfil", blank=True, null=True)

	class Meta:
		ordering = ["created_at"]
		verbose_name_plural = "Experiments"
		verbose_name = "Experiment"

	def __str__(self):
		return str(self.id)
	