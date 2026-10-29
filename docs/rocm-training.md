# RX 9070 XT native Rust GPU training

The user-selected GPU backend is **Rust `tch` 0.24.0 + native LibTorch 2.11.0 / ROCm 7.2**. The actual handwriting VAE uses this native backend by default; `--cpu` selects native CPU training. The retained ORT diagnostic gate exports ONNX proof graphs. No Python trainer or PyO3 is introduced. See [neural model evidence](model-status.md).

## Run

From the repository root on the GPU host:

```sh
nix develop .#rocm
cargo build --locked --offline --manifest-path src/Cargo.toml -p ink-cli --features rocm
./src/target/rocm/debug/ink model gate --backend rocm --out runs/rocm-9070xt-proof
./src/target/rocm/debug/ink check
```

The ROCm shell uses a separate `src/target/rocm` build directory, so it does not overwrite the CPU CLI. `--gpu-index N` explicitly selects a HIP ordinal; omission selects the RX 9070 XT by kernel identity. Selecting the integrated GPU fails. GPU device nodes are hidden in the Codex filesystem sandbox, so hardware execution requires the normal host terminal or an approved execution outside that sandbox. The backend fails instead of falling back to CPU.

`nix run .#training -- train --seconds 120` trains the actual conditional VAE. `model gate --backend rocm` separately exercises the synthetic backend fixture described below; it is not a prerequisite on every training run.

## Physical device and execution proof

HIP returns the generic name `AMD Radeon Graphics` on this host, including for the discrete card. Device names or ordinal zero alone are insufficient evidence. The gate obtains the HIP PCI address and matches it to kernel KFD topology. The accepted signature is AMD vendor `0x1002`, device `0x7550`, `gfx_target_version=120001` (`gfx1201`), and 128 SIMD units. The integrated GPU has a different PCI address and `gfx1036`, and is rejected.

The verified card was HIP ordinal 0 at PCI `0000:03:00.0`. A contract test covers an alternate ordering with the integrated GPU first and ensures explicit selection of it fails.

The gate allocates a 128 × 128 parameter matrix, inputs, targets, intermediate outputs, loss and gradients on the HIP GPU. Rust performs all forward, backward and SGD updates. It checks tensor placement and synchronizes GPU work. ATen calls this device `Cuda(0)` as part of its HIP-compatible API; the PCI/KFD evidence establishes that the hardware is AMD.

Measured result in `reports/rocm-9070xt.json`:

- Eleven real GPU updates; initial loss `4.0`, tenth pre-update loss `0.0720576`.
- Safetensors checkpoint reload is bit-identical on the GPU.
- Resumed update lowers loss from `0.0461169` to `0.0295148` and changes weights.
- Updated and resumed weights are exported to ONNX; Rust ORT inference matches GPU weights with maximum absolute error `0.0` for the gate's identity input.
- The Rust training process loads no `libpython`. Wall time was approximately 2.34 seconds with dependencies already built; this is a backend proof, not a model throughput benchmark.

The proof uses stateless SGD with zero momentum/weight decay; checkpoint parameters plus recorded learning rate and update count fully represent its resume state. Its learning rate scales with the matrix's mean reduction and is not a VAE hyperparameter recommendation. GPU checkpoints, weight JSON, ONNX proof decoders and receipts remain ignored under `runs/` and `reports/`.

## Native package and registration

`src/nix/libtorch-rocm.nix` pins the official PyTorch ROCm 7.2 wheel, SHA256 `8f02a36213288ab5921e277e13f5954cc62afe7cfa899b79d739d4ef2be6510b`. Nix extracts only C++ headers, native libraries and license notices, explicitly removes `libtorch_python.so`, and patches native loader paths. No Python packages or executable are included in the SDK. The native bundle is large (roughly 13 GiB, largely GPU kernels).

Rust's default linker can discard `libtorch_hip.so` as unused, preventing ATen HIP hook registration. The ROCm module explicitly loads it with global symbol visibility and retains it for process lifetime. No global `LD_PRELOAD`, architecture spoofing or CPU fallback is used.

For the optional backend fixture, Python remains confined to the separate offline graph shell: it creates an ONNX MatMul graph from weights already updated and serialized by Rust. It performs no training. The existing inference-only skill executable has no LibTorch/ROCm dependency.

AMD documents [RX 9070 XT ROCm support](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.4.2/). Upstream ORT documents that its [ROCm execution provider was removed after 1.22](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html). AMD's investigated 1.22.1 ROCm training wheel contains gfx1201 kernels but does not export `OrtGetApiBase` in its dynamic symbol table; the user chose the native Rust LibTorch route instead of rebuilding the ORT C API.
