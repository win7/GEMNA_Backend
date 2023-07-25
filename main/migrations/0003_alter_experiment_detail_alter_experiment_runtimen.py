# Generated by Django 4.1 on 2023-07-24 04:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0002_alter_experiment_detail_alter_experiment_email'),
    ]

    operations = [
        migrations.AlterField(
            model_name='experiment',
            name='detail',
            field=models.TextField(blank=True, max_length=100, null=True, verbose_name='Details'),
        ),
        migrations.AlterField(
            model_name='experiment',
            name='runtimen',
            field=models.IntegerField(default=0, verbose_name='Runtime'),
        ),
    ]
