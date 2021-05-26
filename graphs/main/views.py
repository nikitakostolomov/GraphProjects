from django.shortcuts import render, redirect
from .forms import ImageForm
import ctypes, pathlib, json, os.path
from ctypes import cdll
from PIL import Image
from numpy import asarray

def mainpage(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            request.session['image']= serialize(form.instance.image)
    else:
        form = ImageForm()

    img = request.session['image']
    data = {
        'form': form,
        'img': img,
        'is_download_img': os.path.exists('media/'+img)
        
    }
    return render(request, 'main/mainpage.html', context = data)

def serialize(obj):
    return (json.dumps(str(obj)).replace('"', ''))

def segmentation(request):
    img=request.session['image']
    # # Преобразование изображения в массив
    # img = Image.open('media/'+img)
    # data = asarray(img)
    # # Обратно
    # img_out = Image.fromarray(data)

    if request.method == 'POST':
        object_pixels = request.POST.get('object_pixels', False)
        background_pixels = request.POST.get('background_pixels', False)
        print(type(object_pixels))
    data = {
        'img': img,
        'typeofpage' : 'segmenation',
        'is_download_img': os.path.exists('media/'+img)
    }
        
        
    return render(request, 'main/mainpage.html',context = data)


