#!/bin/bash
# sudo apt -y update
# sudo apt -y upgrade
apt -y update
apt -y install python-opencv python3-numpy 
apt -y install python3-opengl 
apt -y install cmake zlib1g-dev libjpeg-dev xvfb ffmpeg 
apt -y install xorg-dev libboost-all-dev libsdl2-dev swig
apt -y install unzip libopenblas-dev liblapack-dev
apt -y install python3-scipy python3-matplotlib python3-yaml
apt -y install libhdf5-serial-dev python-h5py graphviz
apt -y install python3-opencv zip git
apt -y install libsm6 libxext6 libxrender-dev
pip3 install matplotlib
pip3 install pandas
pip3 install pydot-ng
pip3 install gym
pip3 install gym[box2d]
pip3 install gym[atari]

# docker run -u $(id -u):$(id -g) -v $(realpath ~/accord):/accord \
#            -w /accord --gpus '"device=0"'  \
#            -dit tf-atari python3 runc51.py -id 0 -k 4000
# docker run -u $(id -u):$(id -g) -v $(realpath ~/accord):/accord -w /accord --gpus '"device=0"' -it tf-atari