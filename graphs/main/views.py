from django.shortcuts import render
from .forms import ImageForm
import os.path
from .algorithms import start_algorithm, improve_algorithm
from .models import Graph_and_pixels
import json
from PIL import Image
import re


def serialize(obj):
    return json.dumps(str(obj)).replace('"', '')


def convert_str_to_list(str):
    list_of_pairs = []
    list_of_coords = list(str[:-1].split(","))
    for i in range(0, len(list_of_coords) - 1, 2):
        list_of_pairs.append((int(list_of_coords[i]), int(list_of_coords[i+1])))
    print(list_of_pairs)
    print(list(set(list_of_pairs)))
    return list(set(list_of_pairs))

def list_to_str(pixels):
    pixels_in_db=""
    for el in pixels:
        pixels_in_db+=f'{el[0]}'+' '+f'{el[1]}'+' '
    return pixels_in_db[:-1]

def graph_string_to_tuple(graph):
    graph_value=graph.split(" ")
    tuple_graph = ((int(graph_value[0]),int(graph_value[1])),)
    del graph_value[0]
    del graph_value[0]
    dict_graph={}
    for i in range(0,len(graph_value),3):
        dict_graph.update({(int(graph_value[i]),int(graph_value[i+1])):float(graph_value[i+2])})
    tuple_graph+=(dict_graph,)
    return(tuple_graph)

def save_to_database(request, graph, object_pixels, background_pixels, k, tfm, tsm):
    graph_and_pixels = Graph_and_pixels.objects.create(graph=graph, object_pixels =  object_pixels, 
                                    background_pixels = background_pixels, K=k, tfm = tfm, tsm=tsm)
    graph_and_pixels.save()
    request.session['id_graph'] = graph_and_pixels.id_graph

def delete_gaps(pixels):
    pixels=pixels.strip()
    pixels = re.sub(" +", " ", pixels)
    return pixels
        

def mainpage(request, data={}):
    print(data)
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            request.session['image'] = serialize(form.instance.image)
            request.session['image_verify'] = serialize(form.instance.image_verify)
            Graph_and_pixels.objects.all().delete()
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
        Graph_and_pixels.objects.all().delete()
        object_pixels = convert_str_to_list(request.POST.get('object_pixels', ''))
        background_pixels = convert_str_to_list(request.POST.get('background_pixels', ''))
        if request.POST.get('is_eight_neighbors', True) == 'on':
            is_four_neighbors = False
        else:
            is_four_neighbors = True
        graph, k, result_img, tfm, tsm = start_algorithm(img_url, img_verify_url, object_pixels, background_pixels,
                                    is_four_neighbors=is_four_neighbors, 
                                    lyambda=float(request.POST.get('lyambda', '1') if request.POST.get('lyambda', '') != '' else '1'),
                                    sigma=float(request.POST.get('sigma', '0.1') if request.POST.get('sigma', '') != '' else '0.1'))                         
        object_pixels = list_to_str(object_pixels)
        background_pixels = list_to_str(background_pixels)
        save_to_database(request, graph,object_pixels, background_pixels, k, tfm, tsm)
        result_url = "media/imagesresult/imgresult.jpeg"
        result_img.save(result_url)
        data.update({
            'result_img': result_url,
            'object_pixels': object_pixels,
            'background_pixels':background_pixels,
            'after_first_launch': True,
            'can_be_interactive_segmentation': True,
        })
        return mainpage(request, data)

    return mainpage(request)


def interactive_segmentation(request):
    data = {}
    try:
        img_url = 'media/' + request.session['image']
        img_verify_url = 'media/' + request.session['image_verify']
        id_graph = request.session['id_graph']
    except KeyError:
        return mainpage(request)
    if request.method == 'POST':
        object_pixels = convert_str_to_list(request.POST.get('new_object_pixels', ''))
        background_pixels = convert_str_to_list(request.POST.get('new_background_pixels', ''))
        # graph = graph_string_to_tuple(Graph_and_pixels.objects.filter(id_graph=id_graph).values('graph')[0].get('graph'))   
        graph = Graph_and_pixels.objects.filter(id_graph=id_graph).values('graph')[0].get('graph')
        k = Graph_and_pixels.objects.filter(id_graph=id_graph).values('K')[0].get('K')

        graph, result_img, tfm, tsm = improve_algorithm(img_url, img_verify_url, graph, object_pixels, background_pixels, k)  
        # graph = '3 2 1 2 4 2 3 5'
        
        object_pixels = list_to_str(object_pixels)
        background_pixels = list_to_str(background_pixels)
        save_to_database(request, graph,object_pixels, background_pixels, k, tfm=0, tsm=0)
        # save_to_database(request, graph,object_pixels, background_pixels, k, tfm, tsm)

        id_graph = request.session['id_graph']
        object_pixels, background_pixels = '',''
        stop = True
        while stop:
            try:
                object_pixels += ' '+Graph_and_pixels.objects.filter(id_graph=id_graph).values('object_pixels')[0].get('object_pixels')
                background_pixels += ' '+Graph_and_pixels.objects.filter(id_graph=id_graph).values('background_pixels')[0].get('background_pixels')
                id_graph-=1
            except:
                stop = False
        object_pixels = delete_gaps(object_pixels)
        background_pixels = delete_gaps(background_pixels)

        result_url = "media/imagesresult/imgresult.jpeg"
        # result_img.save(result_url)


        data.update({
            'after_first_launch': True,
            'can_be_interactive_segmentation': True,
            'object_pixels': object_pixels,
            'background_pixels':background_pixels,
            'result_img': result_url,
        })

    return mainpage(request,data)

def dropping(request):
    data={}
    if request.method == 'POST':
        Graph_and_pixels.objects.all().delete()
    data.update({
            'can_be_interactive_segmentation': False,
        })
    return mainpage(request,data)

