FROM tensorflow/tensorflow:1.13.2-py3-jupyter

RUN pip3 install scipy>=1.1.0

WORKDIR /tf/neurec

COPY . .

RUN pip3 install --editable .
