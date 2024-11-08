# GEMNA Backend

Based on Django Rest Framework (4.1) with Python (3.10)

## Install requirements (Ubuntu/Linux Mint)
``` bash
# install mini-conda (these four commands download and install the latest 64-bit version of the Linux installer)

# create folder to installer
$ mkdir -p ~/miniconda3
```
output:

![Alt text](/setup_img/conda_mkdir.png)

``` bash
# download installer
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
```
output:

![Alt text](/setup_img/conda_download.png)

``` bash
# install mini-conda
$ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
```
output:

![Alt text](/setup_img/conda_bash.png)

``` bash
# silently install
$ rm -rf ~/miniconda3/miniconda.sh
```
output:

![Alt text](/setup_img/conda_rm.png)

``` bash
# after installing, close and reopen your terminal application or refresh it to activate mini-conda
$ source ~/miniconda3/bin/activate
```
output:

![Alt text](/setup_img/conda_source.png)


## Quick Setup
``` bash
# check nvidia driver version
$ nvidia-smi
```
output:

![Alt text](/setup_img/nvidia_smi.png)

``` bash
# create environment
$ conda env create -f environment_cu121.yml # cuda 12.1
```
output:

![Alt text](/setup_img/install_env1.png)
...
![Alt text](/setup_img/install_env2.png)

``` bash
# activate environment
$ conda activate gemna_3.10
```
output:

![Alt text](/setup_img/activate_env.png)

``` bash
# migrate database (default SQLite)
$ ./manage.py makemigrations
```
output:

![Alt text](/setup_img/makemigrations.png)

``` bash
$ ./manage.py startapp migrate
```
output:

![Alt text](/setup_img/migrate.png)

``` bash
# create super user to login on Django Admin
$ ./manage.py createsuperuser
```
output:

![Alt text](/setup_img/createsuperuser.png)

``` bash
# run server
$ ./manage.py runserver 0.0.0.0:8000
```
output:

![Alt text](/setup_img/runserver.png)

At the end, in the browser type: http://localhost:8000/admin, the web application (Django admin) will be loaded there.

Login

![Alt text](/setup_img/login_drf.png)

Administration

![Alt text](/setup_img/admin_drf.png)

# Frontend and backend integration
Frontend and backend integration depends on the domain or IP of the backend. For example, if the backend is deployed on a server with an IP: **192.168.1.77** and on port: **8000**, then that IP and port must be setup in the [nuxt.config.js](https://github.com/win7/GEMNA_Frontend/blob/main/nuxt.config.js) file of the frontend.

``` bash
...
axios: {
		baseURL: "http://192.168.1.77:8000",
		// progress: false,
		// credentials: true,
		// validateStatus: false
	},
...
```

# User guide
We include an [user guide](https://github.com/win7/GEMNA_Backend/blob/main/GEMNA_User_guide.pdf) to use the GEMNA web applicaction.

