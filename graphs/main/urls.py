from django.contrib import admin
from django.urls import path, include, re_path
from . import views

urlpatterns = [
    path('', views.mainpage,name='mainpage'),
    path('segmentation', views.segmentation,name='segmentation'),

]