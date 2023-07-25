from rest_framework.response import Response
from rest_framework import status

class Resp:
	def __init__(self, data=None, message="OK", flag=True, status=status.HTTP_200_OK):
		self.data = data
		self.message = message
		self.flag = flag
		self.status = status
		
	def send(self):
		result = {
			"data": self.data,
			"message": self.message,
			"flag": self.flag
		}
		return Response(result, status=self.status)
