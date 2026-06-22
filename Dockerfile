FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

SHELL ["/bin/bash", "-lc"]
WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel --break-system-packages && \
    python -m pip install --no-cache-dir jupyterlab --break-system-packages && \
    python -m pip install --no-cache-dir -r /workspace/requirements.txt --break-system-packages

    RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs


COPY . /workspace

CMD ["bash"]