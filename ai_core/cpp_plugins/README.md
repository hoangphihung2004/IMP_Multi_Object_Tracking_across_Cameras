# C++ / CUDA DeepStream Plugins

Thư mục này chứa các plugin C++/CUDA dùng bởi runtime DeepStream.

## Thành phần

- `ocsort/`: OCSort C++ engine và C API để Python gọi qua `ctypes`.
- `yolox_parser/`: custom bounding-box parser cho DeepStream `nvinfer`.
- `preprocess/`: custom CUDA preprocess plugin.

## Vai trò trong runtime local

Runtime Windows + WSL2 + Docker vẫn giữ các plugin này:

- YOLOX chạy qua DeepStream/TensorRT.
- OCSort C++ xử lý tracking.
- Python chỉ điều phối pipeline và metadata.

## Build

Dockerfile tự build các plugin này khi tạo image:

```bash
cd cpp_plugins/<plugin>
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

Các file `.so` sau khi build được copy vào:

```text
cpp_plugins/lib/
```
