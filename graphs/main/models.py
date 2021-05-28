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
