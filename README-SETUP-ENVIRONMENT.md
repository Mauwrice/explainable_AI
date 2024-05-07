# The environment
## Hardware and Software Configuration
Experiments for this paper have been conducted using [Docker](https://docs.docker.com/get-started/) as a platform. The version is: `23.0.6`

The underlying host runs the Linux distribution `Ubuntu 18.04.6 LTS`.

The GPU is a NVIDIA `TITAN RTX`, with the following CUDA configuration:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Feb_14_21:12:58_PST_2021
Cuda compilation tools, release 11.2, V11.2.152
Build cuda_11.2.r11.2/compiler.29618528_0
```
``` 
NVIDIA-SMI 460.32.03
```

The utilized CPU is a `Intel Core i9-9820X @ 3.30GHz` with `64GB` of memory available.

## How to start the docker container with the environment
The docker images are published on Dockerhub and can be started on a server. The images are available on https://hub.docker.com/repository/docker/nordar/stroke_perfusion/general

This particular study utilized the image `nordar/stroke_perfusion:2.2.0` to build the container for training. The image `nordar/stroke_perfusion:2.2.0-publish` includes all python dependencies that were installed into the `2.2.0` version.

To create a container from the `nordar/stroke_perfusion:2.2.0-publish` image one can use the following command on the GPU workstation/server:

```docker
# This is only neccessary if the authentication token under ~/.docker/config.json is expired
docker login

# Download the image from hub.docker.com and start a container
docker run \
    --rm \
    --mount type=bind,source="$(pwd)",target=/tf/notebooks \
    -p 8383:8888 \
    --gpus device=0 \
    --name xai-notebook \
    nordar/stroke_perfusion:2.2.0-publish
```
The following output will be presented:
```
[I 10:01:28.542 NotebookApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret
[I 10:01:28.731 NotebookApp] Serving notebooks from local directory: /tf
[I 10:01:28.731 NotebookApp] The Jupyter Notebook is running at:
[I 10:01:28.731 NotebookApp] http://3bd3124903c7:8888/?token=6b42a391ed0eac976b3bab966aa3b5131791589298f5abf4
[I 10:01:28.731 NotebookApp]  or http://127.0.0.1:8888/?token=6b42a391ed0eac976b3bab966aa3b5131791589298f5abf4
[I 10:01:28.731 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 10:01:28.735 NotebookApp]

    To access the notebook, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/nbserver-1-open.html
    Or copy and paste one of these URLs:
        http://3bd3124903c7:8888/?token=6b42a391ed0eac976b3bab966aa3b5131791589298f5abf4
     or http://127.0.0.1:8888/?token=6b42a391ed0eac976b3bab966aa3b5131791589298f5abf4
```

One can now access the `jupyter notebook` instance under the address `http://127.0.0.1:8888/?token=6b42a391ed0eac976b3bab966aa3b5131791589298f5abf4` where the `token` part of the url will change with every startup of the image. Replace `127.0.0.1` in the url with the workstations IP address or with the workstations domain name and replace the port `8888` with the port specified in the command (``-p 8383:8888``), so in this case `8383`. 

For example: `http://gpu-server.uni.edu:8383/?token=6b42a3.....`


## Docker Command Explained
Find the complete information in the docker documentaion. https://docs.docker.com/engine/reference/run/

The utilized flags are explained here:

| Docker command                                     | Explaination|
|---------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| docker run                                              | Start a docker container from an image.                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| --rm                                                    | Remove the docker container after the process (in this case jupyter notebook) has stopped. This can be ommited to persist the container and changes within.                                                                                                                                                                                                                                                                                                                          |
| --mount type=bind,source="$(pwd)", target=/tf/notebooks  | This mounts a local directory into the docker container. In this case, "$(pwd)" will resolve to the directory where this command was executed from.  Alternatively one can also put in paths to data directories or project directories. It is also possible to mount multiple directories into the container.  `source` refers to the local directory and `target` refers to the docker container. More information can be found here: https://docs.docker.com/storage/bind-mounts/ |
| -p 8383:8888                                            | This port will be forwarded from the docker container to the local operating system. The `jupyter notebook` runs on port `8888` within the docker container. Port `8383` is the port that workstation will forward to the containers port.  This means the port defined on the left side is the one you need to access in the browser so to access the jupyter instance within the container. The port on the right shall only be modified if one knows what they are doing.         |
| --gpus device=0                                         | The enabled GPUs where the number indicates an index starting with 0.                                                                                                                                                                                                                                                                                                                                                                                                                |
| --name xai-notebook                                     | The name of the container created. Helps with assigning containers to uses. Might not be relevant if one chooses to `--rm` the containers after closing jupyter.                                                                                                                                                                                                                                                                                                                     |
| nordar/stroke_perfusion:2.2.0-publish                   | The name of the image used to start a container. Contains the following components **``<docker hub repository name>``/``<image name>``:``<image tag>``**                                                                                                                                                                                                                                                                                                                                             |