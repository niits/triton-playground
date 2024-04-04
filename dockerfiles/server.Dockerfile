FROM nvcr.io/nvidia/tritonserver:24.03-py3

RUN apt update \
    && apt-get install libopencv-dev libopencv-core-dev -y \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*