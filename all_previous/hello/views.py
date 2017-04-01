<<<<<<< HEAD
from django.shortcuts import render
from django.http import HttpResponse
import requests

from .models import Greeting

# Create your views here.
'''
def index(request):
    # return HttpResponse('Hello from Python!')
    return render(request, 'index.html')
'''

def index(request):
    r = requests.get('http://httpbin.org/status/418')
    print r.text
    return HttpResponse('<pre>' + r.text + '</pre>')


def db(request):

    greeting = Greeting()
    greeting.save()

    greetings = Greeting.objects.all()

    return render(request, 'db.html', {'greetings': greetings})

=======
from django.shortcuts import render
from django.http import HttpResponse
import requests

from .models import Greeting

# Create your views here.
'''
def index(request):
    # return HttpResponse('Hello from Python!')
    return render(request, 'index.html')
'''

def index(request):
    r = requests.get('http://httpbin.org/status/418')
    print r.text
    return HttpResponse('<pre>' + r.text + '</pre>')


def db(request):

    greeting = Greeting()
    greeting.save()

    greetings = Greeting.objects.all()

    return render(request, 'db.html', {'greetings': greetings})

>>>>>>> dd3dd770488c8ba2e17cc5af9b2a6fc91623a108
