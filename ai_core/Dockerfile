FROM nvcr.io/nvidia/deepstream:7.0-triton-multiarch

# =========================================================
# Environment
# =========================================================
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/workspace/ai_core/cpp_plugins/lib:${LD_LIBRARY_PATH}

WORKDIR /workspace/ai_core

# =========================================================
# 1. System Dependencies (least frequent change → cached)
# =========================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build Tools
    wget git cmake curl build-essential pkg-config \
    # Python Core
    python3-pip python3-dev \
    # GStreamer Python Bindings (pyds/gi)
    python3-gi python3-gst-1.0 gir1.2-gst-plugins-base-1.0 \
    libcairo2-dev libgirepository1.0-dev \
    # OpenCV (C++ headers for OCSort + Python bindings)
    python3-opencv libopencv-dev \
    # Eigen3 (OCSort KalmanFilter)
    libeigen3-dev \
    # CUDA tools (nvjpegdec needs this)
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# 2. PYDS — DeepStream Python Bindings (DS 7.0 / JetPack 6)
# =========================================================
RUN pip3 install --upgrade pip \
    && wget -q https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.11/pyds-1.1.11-py3-none-linux_aarch64.whl \
    && pip3 install pyds-1.1.11-py3-none-linux_aarch64.whl \
    && rm pyds-1.1.11-py3-none-linux_aarch64.whl

# =========================================================
# 3. Python Dependencies
# =========================================================
COPY requirements.txt .

RUN pip3 install --upgrade pip setuptools wheel \
    && pip3 install cython ninja \
    && pip3 install --no-cache-dir -r requirements.txt

# =========================================================
# 4. Copy Application Code
# =========================================================
COPY . .

# =========================================================
# 5. Directories (mount points for volumes)
# =========================================================
RUN mkdir -p pretrained videos logs

# =========================================================
# 6. Build C++ Plugins
# =========================================================
# 6a. YOLOX Custom BBox Parser → libnvdsinfer_custombboxparser_yolox.so
RUN cd cpp_plugins/yolox_parser \
    && mkdir -p build && cd build \
    && cmake .. && make -j$(nproc) \
    && mkdir -p /workspace/ai_core/cpp_plugins/lib \
    && cp *.so /workspace/ai_core/cpp_plugins/lib/

# 6b. YOLOX CUDA Preprocess → libyolox_preprocess.so
RUN cd cpp_plugins/preprocess \
    && mkdir -p build && cd build \
    && cmake .. && make -j$(nproc) \
    && cp *.so /workspace/ai_core/cpp_plugins/lib/

# 6c. OCSort C-API Tracker → libocsort_api.so
RUN cd cpp_plugins/ocsort \
    && mkdir -p build && cd build \
    && cmake .. && make -j$(nproc) \
    && cp *.so /workspace/ai_core/cpp_plugins/lib/

# =========================================================
# 7. Install YOLOX in editable mode (Python C++ extensions)
# =========================================================
RUN pip3 install --no-build-isolation -e .

# =========================================================
# 8. Verify Builds
# =========================================================
RUN echo "=== Checking .so libraries ===" \
    && ls -la /workspace/ai_core/cpp_plugins/lib/

# =========================================================
# 9. Entrypoint
# =========================================================
CMD ["python3", "main.py"]
