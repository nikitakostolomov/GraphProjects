from django.db import models


class Image(models.Model):
    class Meta:
        db_table = 'image'

    image = models.ImageField(upload_to='images')
    image_verify = models.ImageField(upload_to='imagesverify')


class Result(models.Model):
    class Meta:
        db_table = 'result'

    image_result = models.ImageField(upload_to='imagesresult')

class Graph_and_pixels(models.Model):
    id_graph = models.AutoField(primary_key=True)
    graph = models.TextField()
    object_pixels = models.TextField()
    background_pixels = models.TextField()
    K = models.FloatField()



