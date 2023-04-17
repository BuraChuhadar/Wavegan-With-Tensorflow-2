FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
        tensorflow-gpu==2.11.0 \
        librosa \
        numpy==1.23.5

# Set up the working directory
WORKDIR /app

COPY . /app

CMD ["bash"]
