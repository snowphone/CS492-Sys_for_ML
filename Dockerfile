FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04 
MAINTAINER Jongse Park

RUN apt-get update  
RUN apt-get install -y build-essential wget python3 python3-pip python3-dev git libssl-dev \
					   vim tmux wget autoconf automake libtool curl make g++ unzip language-pack-en \
					   ffmpeg libsm6 libxrender-dev

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install setuptools numpy==1.16.4 opencv-python cython tensorflow-gpu==1.15.2 networkx
RUN python3 -m pip uninstall -y setuptools
RUN python3 -m pip install setuptools

RUN mkdir /home/cs492
COPY conf/.tmux.conf /root
COPY conf/.bashrc /root
COPY conf/.inputrc /root 
RUN mkdir /root/.vim
COPY conf/.vim /root/.vim
COPY conf/.vimrc /root
WORKDIR /home/cs492/projects/cs492-projects
