# Enabled Rust ONNX Runtime training

RX 9070 XT GPU training is also enabled via the user-selected [native Rust LibTorch/ROCm backend](rocm-training.md); this document describes the retained CPU ORT path.

The backend now uses Microsoft's released **CPU on-device training 1.19.2** library, packaged reproducibly by Nix. Both `ort` and `ort-sys` are pinned to **2.0.0-rc.8**, which targets **C API 19**. The previous rc.10 binding targeted API 22 and could not use this released training binary. No API constants or upstream Rust bindings are patched.

## Reproduce

From the repository root:

```sh
nix develop .
cargo build --locked --offline --manifest-path src/Cargo.toml -p ink-cli
./src/target/debug/ink model gate --out runs/ort-training-enabled
```

Use a fresh gate directory to rebuild graph artifacts. The default Rust shell exposes `ORT_DYLIB_PATH` for the training-enabled library and contains no Python. Rust invokes `nix develop .#graph` only for offline artifact creation. You can inspect the pinned packages independently with `nix build .#training-runtime` and `nix build .#graph-tools`.

## Pinned packages and isolation

[Microsoft.ML.OnnxRuntime.Training 1.19.2](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Training/1.19.2) supplies the Linux x86_64 CPU shared libraries. Its NuGet archive SHA256 is `347dba6221449a9a7f3bfcff82c20fb3acfed3f3f3c545508863c33bcf3093e1`. Nix extracts only Linux libraries and license notices, then patches native loader paths.

The matching [onnxruntime-training-cpu 1.19.2](https://pypi.org/project/onnxruntime-training-cpu/1.19.2/) CPython 3.12 wheel has SHA256 `4c8d4806c5070dc6c175e1a968e45d948588edbfe590b5ef9d42d3511d80e220`. Its isolated graph environment uses a hash-pinned ONNX 1.17.0 wheel and import checks. The upstream training namespace eagerly imports Torch optimizer helpers; the Nix package narrows that namespace to `artifacts` only. This removes the unrelated Python optimizer dependency without changing the graph builder or native training implementation. Python performs no optimizer steps or dataset preprocessing.

All package definitions live in `src/nix/ort-training.nix`. Downloads, graphs, checkpoints and exported proof decoders are ignored by Git. Neural skill exports use a pure Rust inference binary and do not acquire ORT, LibTorch or Python.

## Evidence and scope

The real Rust gate trains a one-weight MatMul model toward target 2. It executes ten optimizer updates, confirms loss decreases, exports updated weights, saves/reloads the checkpoint, verifies identical predictions after reload, performs an eleventh update and verifies the resumed decoder improves again. Failure reports preserve the actual completed update count.

Measured result: 11 weight updates; initial loss `4.0`, tenth pre-update loss `0.0720576`; exported prediction `1.7852516`, identical after reload; resumed prediction `1.8282013` toward target `2.0`. The complete gate took approximately 1.14 seconds with dependencies already built.

The private receipt is `reports/ort-training-enabled.json`; artifacts are under `runs/ort-training-enabled/`. This proves **backend training, checkpoint restore and updated ONNX export**, not a trained handwriting VAE. Actual handwriting training is implemented separately with native Rust LibTorch; see [model evidence](model-status.md).

Microsoft documents both [on-device training packages](https://onnxruntime.ai/docs/install/) and the alternative [source-build training flags](https://onnxruntime.ai/docs/build/training.html). The released CPU package avoids a large C++ rebuild while preserving the Rust/Python boundary.
