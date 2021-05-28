from django.shortcuts import render
from .forms import ImageForm
import os.path
from .algorithms import start_algorithm
from .models import Graph_and_pixels
import json


def serialize(obj):
    return json.dumps(str(obj)).replace('"', '')


def convert_str_to_list(str):
    list_of_pairs = []
    list_of_coords = list(str.split(","))
    for i in range(0, len(list_of_coords) - 1, 2):
        list_of_pairs.append((int(list_of_coords[i + 1]), int(list_of_coords[i])))
    return list_of_pairs


def mainpage(request, data={}):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            request.session['image'] = serialize(form.instance.image)
            request.session['image_verify'] = serialize(form.instance.image_verify)
    else:
        form = ImageForm()
    try:
        img_session = request.session['image']
        is_download_img = os.path.exists('media/' + img_session)
    except KeyError:
        is_download_img = False
        img_session = None
    data.update({
        'img': img_session,
        'is_download_img': is_download_img,
        'form': form,
    })
    return render(request, 'main/mainpage.html', context=data)


def segmentation(request):
    data = {}
    # img_out = Image.fromarray(data)
    try:
        img_url = 'media/' + request.session['image']
        img_verify_url = 'media/' + request.session['image_verify']
    except KeyError:
        return mainpage(request)
    if request.method == 'POST':
        object_pixels = convert_str_to_list(request.POST.get('object_pixels', ''))
        background_pixels = convert_str_to_list(request.POST.get('background_pixels', ''))
        if request.POST.get('is_eight_neighbors', True) == 'on':
            is_four_neighbors = False
        else:
            is_four_neighbors = True
        result_img = start_algorithm(img_url, img_verify_url, object_pixels, background_pixels,
                                     is_four_neighbors=is_four_neighbors, lyambda=float(request.POST.get('lyambda', 1)),
                                     sigma=float(request.POST.get('sigma', 0.1)))
        result_url = "media/imagesresult/imgresult.jpeg"
        result_img.save(result_url)
        data.update({
            'result_img': result_url,
        })

        return mainpage(request, data)

    return mainpage(request)


def interactive_segmentation(request):
    data = {}
    # graph_and_pixels = Graph_and_pixels.objects.create(graph=graph, object_pixels = object_pixels, background_pixels = background_pixels, K=0.1)
    # graph_and_pixels.save()
    # request.session['id_graph'] = graph_and_pixels.id_graph
    try:
        img_url = 'media/' + request.session['image']
        img_verify_url = 'media/' + request.session['image_verify']
    except KeyError:
        return mainpage(request)
    if request.method == 'POST':
        id_graph = request.session['id_graph']
        graph = Graph_and_pixels.objects.filter(id_graph=id_graph).values('graph')[0].get('graph')
        object_pixels = Graph_and_pixels.objects.filter(id_graph=id_graph).values('object_pixels')[0].get(
            'object_pixels')
        background_pixels = Graph_and_pixels.objects.filter(id_graph=id_graph).values('background_pixels')[0].get(
            'background_pixels')
        K = Graph_and_pixels.objects.filter(id_graph=id_graph).values('K')[0].get('K')
