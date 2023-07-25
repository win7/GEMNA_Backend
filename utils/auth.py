from rest_framework.response import Response
from rest_framework import status

from Configuration.models import *
from Maintenance.models import *

def getUser(request):
	if request.user.is_authenticated and request.user.is_active:
		user = request.user
		return user.id
	else:
		return None

def getCompany(request):
	if request.user.is_authenticated and request.user.is_active:
		try:
			company = request.user.profile.company_id
			return company.company_id
		except Company.DoesNotExist:
			return None
	else:
		return None


def getTeacher(request):
	if request.user.is_authenticated and request.user.is_active:
		try:
			person = request.user.person
			teacher = Teacher.objects.get(person_id=person.person_id)
			return teacher.id
		except Teacher.DoesNotExist:
			return None
	else:
		return None

def getStudent(request):
	if request.user.is_authenticated and request.user.is_active:
		try:
			person = request.user.person
			student = Student.objects.get(person_id=person.person_id)
			return student.id
		except Student.DoesNotExist:
			return None
	else:
		return None