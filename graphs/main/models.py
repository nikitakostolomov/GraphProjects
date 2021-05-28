from django.db import models

class Image(models.Model):
    image = models.ImageField(upload_to='images')
    image_verify = models.ImageField(upload_to='imagesverify')
class Result(models.Model):
    image_result = models.ImageField(upload_to='imagesresult')
