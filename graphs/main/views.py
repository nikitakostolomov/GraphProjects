from django.shortcuts import render, redirect
from .forms import ImageForm
import ctypes, pathlib, json, os.path, cv2, io
from ctypes import cdll
from PIL import Image
from numpy import asarray
import itertools
from .algorithms import graph_by_image
from .algorithms import start_algorithm
from .models import Result
import numpy as np


def serialize(obj):
    return (json.dumps(str(obj)).replace('"', ''))

def convert_str_to_list(str):
    list_of_pairs=[]
    list_of_coords=list(str.split(","))
    for i in range(0,len(list_of_coords)-1,2):
        list_of_pairs.append((int(list_of_coords[i+1]),int(list_of_coords[i])))
    return list_of_pairs


def mainpage(request, data={}):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            request.session['image']= serialize(form.instance.image)
            request.session['image_verify'] = serialize(form.instance.image_verify)
    else:
        form = ImageForm()
    try:
        img_session = request.session['image']
        is_download_img = os.path.exists('media/'+img_session)
    except KeyError:
        is_download_img = False
        img_session = None
    data.update({
        'img': img_session,
        'is_download_img': is_download_img,
        'form': form,
    })
    return render(request, 'main/mainpage.html', context = data)

def segmentation(request):
    data={}
    # img_out = Image.fromarray(data)
    try:
        img_url = 'media/'+request.session['image']
        img_verify_url = 'media/'+request.session['image_verify']
    except KeyError:
        return mainpage(request)
    if request.method == 'POST':
        object_pixels = convert_str_to_list(request.POST.get('object_pixels', ''))
        background_pixels = convert_str_to_list(request.POST.get('background_pixels', ''))
        result_img = start_algorithm(img_url, img_verify_url, object_pixels, background_pixels, is_four_neighbors=True,  lyambda = 1, sigma = 1)
        result_url="media/imagesresult/imgresult.jpeg"
        result_img.save(result_url)
        data.update({
        'result_img': result_url,
        })
        return mainpage(request,data)

    return mainpage(request)
    




