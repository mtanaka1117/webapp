docker container run --rm --gpus all -v /home/srv-admin/docker:/docker -e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) -it nvidia/cuda:12.3.1-base-ubuntu20.04 /bin/bash
