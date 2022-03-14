FROM nvidia/cuda:11.2.1-base

WORKDIR /experiments_motion
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install git
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
RUN  apt-get update && apt-get install -y python3-opencv
# By copying over requirements first, we make sure that Docker will cache
# our installed requirements rather than reinstall them on every build
COPY requirements.txt /experiments_motion/requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install notebook

ENV PYTHONPATH=.