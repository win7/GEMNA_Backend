# Generated by Django 4.1 on 2023-10-09 19:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0012_remove_experiment_threshold_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='experiment',
            name='data_variation',
            field=models.CharField(choices=[('', 'none'), ('str', 'str'), ('dyn', 'dyn')], default='', max_length=4, verbose_name='Dataset variation'),
        ),
    ]
