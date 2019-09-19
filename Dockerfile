FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

RUN pip3 install scipy>=1.1.0 importlib_resources==1.0.2

WORKDIR /tf/neurec

COPY . .

RUN pip3 install --editable .