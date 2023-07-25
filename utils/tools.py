"""import urllib.request
def fetch_url(url,method,data):
    if method=="GET":
        response = urllib.request.urlopen(url,data=data)
    else:
        # Usually a POST
        response = urllib.request.urlopen(url,data=data) 
    # Fetch the content and response code
    [content, response_code] = (response.read(), response.code)
    # Check if the server responded with a success code (200)
    if (response_code == 200):
        return content
    else:
        return response_code"""

from socket import *
from datetime import datetime, date, time, timedelta
import time
import math

def ClientSocket(host, port, data):
    buf = 1024
    result = 'error'
    try:
        addr = (host, port)
        clientsocket = socket(AF_INET, SOCK_STREAM)
        clientsocket.settimeout(5)
        clientsocket.connect(addr)
        #clientsocket.settimeout(None)
        result = clientsocket.recv(buf).encode('utf-8')
        clientsocket.send(data.encode('utf-8'))
        k = 0
        while k<25:
            try:
                result = clientsocket.recv(buf).encode('utf-8')
                print(str(k)+' --> '+str(result))
                if 'ok' in str(result) or 'error' in str(result):
                    break
            except Exception as e:
                result = 'error error'
            k+=1  
    except Exception as e:
        result = 'error error error'
    clientsocket.close()
    print('close')
    return result 

def FormatControlPoint(e):
    return "{:0>2d}".format(e)

def AddMinute(time_, min):
    time_ = datetime.strptime(str(time_), '%H:%M:%S')
    new_time = time_ + timedelta(seconds=min*60)
    return new_time.strftime("%H:%M:%S")

def DeductMinute(time_, min):
    time_ = datetime.strptime(str(time_), '%H:%M:%S')
    new_time = time_ - timedelta(seconds=min*60)
    return new_time.strftime("%H:%M:%S")
    
def DeductTime(time1, time2):
    time1 = datetime.strptime(str(time1), '%H:%M:%S')
    time2 = datetime.strptime(str(time2), '%H:%M:%S')
    if not "00:00:00" in str(time2):
        if time1 > time2:
            dex = time1-time2
            return (int(-1*dex.seconds/60))
        else:
            dex = time2-time1
            return (int(dex.seconds/60))
    else:
        return 0
    
def StrToBool(v):
    return v.lower() in ("yes", "true", "t", "1")

def DistanceCoordinates(latitude1, longitude1, latitude2, longitude2):
    latitude1 = latitude1 * math.pi / 180
    longitude1 = longitude1 * math.pi / 180
    latitude2 = latitude2 * math.pi / 180
    longitude2 = longitude2 * math.pi / 180
    return round(6378137 * (math.acos(math.cos(latitude1) * math.cos(latitude2) * math.cos(longitude2-longitude1) + math.sin(latitude1) * math.sin(latitude2))))