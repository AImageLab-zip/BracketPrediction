FROM pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel

WORKDIR /workspace
ENV PYTHONPATH=.

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends xvfb xauth x11-utils && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install -r requirements.txt
