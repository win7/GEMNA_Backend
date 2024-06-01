# Generated by Django 4.1 on 2024-06-01 00:17

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0018_alter_experiment_method'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='experiment',
            name='control',
        ),
        migrations.AddField(
            model_name='experiment',
            name='controls',
            field=models.CharField(default=django.utils.timezone.now, max_length=64, verbose_name='Controls'),
            preserve_default=False,
        ),
    ]