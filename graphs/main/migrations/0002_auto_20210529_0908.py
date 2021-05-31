# Generated by Django 3.2.3 on 2021-05-29 09:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Graph_and_pixels',
            fields=[
                ('id_graph', models.AutoField(primary_key=True, serialize=False)),
                ('graph', models.TextField()),
                ('object_pixels', models.TextField()),
                ('background_pixels', models.TextField()),
                ('K', models.FloatField()),
            ],
        ),
        migrations.AlterModelTable(
            name='result',
            table='result',
        ),
    ]
