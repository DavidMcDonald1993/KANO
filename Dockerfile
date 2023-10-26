# miniconda preferred to anaconda
FROM continuumio/miniconda3 AS build

# install any required system dependencies
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    ca-certificates \
    build-essential \
    cmake \
    wget \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# set working directory to container root
WORKDIR /

# install python dependencies
COPY ./environment.yml /environment.yml
RUN conda env update --name myenv --file /environment.yml
RUN rm /environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# all pip dependencies
COPY ./requirements.txt /requirements.txt
RUN pip install requirements.txt

# Make RUN commands use the base environment:
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# install conda-pack into base env 
RUN conda install -c conda-forge conda-pack

# Use conda-pack to create a standalone enviornment
# in /venv:
RUN conda-pack -n myenv -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

# We've put venv in same path it'll be in final image,
# so now fix up paths:
RUN /venv/bin/conda-unpack

# The runtime-stage image; we can use Debian as the
# base image since the Conda env also includes Python
# for us.
FROM debian:buster AS runtime

# Copy /venv from the previous stage:
COPY --from=build /venv /venv

# copy required directories 
COPY chemprop /chemprop
# TODO: only copy required model(s)
COPY dumped /dumped 

# copy required files
# python source code 
COPY inference_cpu.py entrypoint.sh /

# bind mount run dir

ENTRYPOINT [ "/entrypoint.sh" ]