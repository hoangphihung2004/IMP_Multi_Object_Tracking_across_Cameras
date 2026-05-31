set -e

# Ensure onnxscript is installed for PyTorch ONNX exporter
python3 -c "import onnxscript" 2>/dev/null || pip3 install onnxscript || true

# --- Always recompile C++ plugins to pick up source changes ---
echo "[Entrypoint] Compiling C++ plugins..."

# Compile OCSort
echo "--- Compiling OCSort ---"
cd /workspace/ai_core/cpp_plugins/ocsort
rm -rf build && mkdir -p build && cd build && cmake .. && make -j$(nproc)

# Compile YOLOX Parser
echo "--- Compiling YOLOX Parser ---"
cd /workspace/ai_core/cpp_plugins/yolox_parser
rm -rf build && mkdir -p build && cd build && cmake .. && make -j$(nproc)

# Compile Custom Preprocess
echo "--- Compiling Preprocess ---"
cd /workspace/ai_core/cpp_plugins/preprocess
rm -rf build && mkdir -p build && cd build && cmake .. && make -j$(nproc)

# Copy compiled libraries to lib folder
mkdir -p /workspace/ai_core/cpp_plugins/lib
cp /workspace/ai_core/cpp_plugins/ocsort/build/*.so /workspace/ai_core/cpp_plugins/lib/ 2>/dev/null || true
cp /workspace/ai_core/cpp_plugins/yolox_parser/build/*.so /workspace/ai_core/cpp_plugins/lib/ 2>/dev/null || true
cp /workspace/ai_core/cpp_plugins/preprocess/build/*.so /workspace/ai_core/cpp_plugins/lib/ 2>/dev/null || true

echo "[Entrypoint] Compilation complete!"
cd /workspace/ai_core

echo "[Entrypoint] Executing command: $@"
exec "$@"
