from django.shortcuts import render, redirect
from .forms import ImageForm
import ctypes, pathlib, json, os.path
from ctypes import cdll
from PIL import Image
from numpy import asarray
import itertools

def serialize(obj):
    return (json.dumps(str(obj)).replace('"', ''))

def convert_str_to_list(str):
    list_of_pairs=[]
    list_of_coords=list(str.split(","))
    for i in range(0,len(list_of_coords)-1,2):
        list_of_pairs.append((int(list_of_coords[i]),int(list_of_coords[i+1])))
    return list_of_pairs


def mainpage(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            request.session['image']= serialize(form.instance.image)
    else:
        form = ImageForm()

    img_session = request.session['image']
    data = {
        'img': img_session,
        'is_download_img': os.path.exists('media/'+img_session)
        
    }
    return render(request, 'main/mainpage.html', context = data)

def segmentation(request):
    # img_out = Image.fromarray(data)

    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            request.session['image']= serialize(form.instance.image)
        else:
            form = ImageForm()
        # object_pixels = convert_str_to_list(request.POST.get('object_pixels', ''))
        # background_pixels = convert_str_to_list(request.POST.get('background_pixels', ''))
        # img = Image.open('media/'+img_session)
        # data = asarray(img)

    img_session=request.session['image']
    data = {
        'img': img_session,
        'is_download_img': os.path.exists('media/'+img_session)
    }
          
    return render(request, 'main/mainpage.html',context = data)




