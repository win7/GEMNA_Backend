# GEMNA Backend

Based on Django Rest Framework (4.1) with Python (3.10)

## Install requirements (Ubuntu/Linux Mint)
``` bash
# install anaconda
$ mkdir -p ~/miniconda3
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
$ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
$ rm -rf ~/miniconda3/miniconda.sh
$ ~/miniconda3/bin/conda init bash
$ ~/miniconda3/bin/conda init zsh

# create environment
$ conda env export > environment_cu118.yml # cuda 11.8
or
$ conda env export > environment_cu121.yml # cuda 12.1
```

## Quick Setup
``` bash
# migrate database (default SQLite)
$ ./manage.py makemigrations
$ ./manage.py startapp migrate

# run server
$ ./manage.py runserver 0.0.0.0:8000
```
