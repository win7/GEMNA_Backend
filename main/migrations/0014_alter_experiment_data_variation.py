# Generated by Django 4.1 on 2023-10-09 19:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0013_alter_experiment_data_variation'),
    ]

    operations = [
        migrations.AlterField(
            model_name='experiment',
            name='data_variation',
            field=models.CharField(choices=[('none', 'none'), ('str', 'str'), ('dyn', 'dyn')], default='none', max_length=4, verbose_name='Dataset variation'),
        ),
    ]
