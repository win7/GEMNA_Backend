# Generated by Django 4.1 on 2023-07-23 22:07

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Experiment',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False, verbose_name='Id')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='Created')),
                ('updated_at', models.DateTimeField(auto_now=True, verbose_name='Updated')),
                ('method', models.CharField(choices=[('m1', 'DGI'), ('m2', 'VGAE')], default='m1', max_length=2, verbose_name='Method')),
                ('data_variation', models.CharField(choices=[('o1', 'none'), ('o2', 'str'), ('o3', 'dyn')], default='o1', max_length=2, verbose_name='Dataset variation')),
                ('dimension', models.IntegerField(default=3, verbose_name='Dimension')),
                ('raw_data', models.FileField(upload_to='raw_data/', verbose_name='Raw data')),
                ('runtimen', models.IntegerField(default=0, verbose_name='runtime')),
                ('email', models.EmailField(blank=True, max_length=254, null=True, verbose_name='Email')),
                ('detail', models.TextField(max_length=100, verbose_name='Detalles')),
            ],
            options={
                'verbose_name': 'Experiment',
                'verbose_name_plural': 'Experiments',
                'ordering': ['created_at'],
            },
        ),
    ]
