from rest_framework import serializers
from main.models import *

class ExperimentSerializer(serializers.ModelSerializer):
	class Meta:
		model = Experiment
		fields = "__all__"