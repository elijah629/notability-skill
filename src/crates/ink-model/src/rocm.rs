use anyhow::{Context, Result, ensure};
use ink_corpus::{file_hash, save};
use serde_json::{Value, json};
use std::{
    ffi::{CStr, c_char, c_int},
    fs,
    path::Path,
    process::{Command, Stdio},
    sync::OnceLock,
    time::Instant,
};
use tch::{Device, Kind, Tensor, nn, nn::OptimizerConfig};

// ATen's HIP hooks are registered by this library's static initializers.
// Keep it loaded for process lifetime; its hooks outlive individual tensors.
static HIP_BACKEND: OnceLock<libloading::os::unix::Library> = OnceLock::new();
fn register_hip(libtorch: &Path) -> Result<()> {
    if HIP_BACKEND.get().is_none() {
        let library = unsafe {
            libloading::os::unix::Library::open(
                Some(libtorch.join("lib/libtorch_hip.so")),
                libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
            )
        }?;
        let _ = HIP_BACKEND.set(library);
    }
    Ok(())
}
const SIZE: i64 = 128;
const LEARNING_RATE: f64 = 0.1 * (SIZE * SIZE) as f64;

#[derive(Clone, Debug, serde::Serialize)]
struct GpuIdentity {
    hip_name: String,
    pci_bus_id: String,
    vendor_id: u64,
    device_id: u64,
    gfx_target_version: u64,
    simd_count: u64,
}
impl GpuIdentity {
    fn is_9070xt(&self) -> bool {
        self.vendor_id == 0x1002
            && self.device_id == 0x7550
            && self.gfx_target_version == 120001
            && self.simd_count == 128
    }
}
/// HIP may expose a generic product name. Match its PCI address to kernel KFD
/// topology, which distinguishes the gfx1201 discrete GPU from the gfx1036 iGPU.
fn identity(libtorch: &Path, index: usize) -> Result<GpuIdentity> {
    type Query = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
    let library = unsafe { libloading::Library::new(libtorch.join("lib/libamdhip64.so")) }?;
    let query = |symbol: &[u8]| -> Result<String> {
        // Both HIP C functions accept (buffer, byte length, device ordinal).
        let function = unsafe { library.get::<Query>(symbol) }?;
        let mut buffer = [0 as c_char; 256];
        let status = unsafe {
            function(
                buffer.as_mut_ptr(),
                buffer.len() as c_int,
                index.try_into()?,
            )
        };
        ensure!(
            status == 0,
            "HIP identity query failed ({status}); check GPU access outside sandbox"
        );
        ensure!(buffer.contains(&0), "HIP identity was not terminated");
        Ok(unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_str()?
            .to_owned())
    };
    let hip_name = query(b"hipDeviceGetName\0")?;
    let pci_bus_id = query(b"hipDeviceGetPCIBusId\0")?;
    let parts: Vec<_> = pci_bus_id.split([':', '.']).collect();
    ensure!(parts.len() == 4, "invalid HIP PCI address");
    let numbers = parts
        .iter()
        .map(|s| u64::from_str_radix(s, 16))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let location = (numbers[1] << 8) | (numbers[2] << 3) | numbers[3];
    for node in fs::read_dir("/sys/class/kfd/kfd/topology/nodes")? {
        let text = fs::read_to_string(node?.path().join("properties"))?;
        let fields = text
            .lines()
            .filter_map(|line| line.split_once(' '))
            .map(|(k, v)| Ok((k, v.parse::<u64>()?)))
            .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
        if fields.get("domain") == Some(&numbers[0]) && fields.get("location_id") == Some(&location)
        {
            let field = |key: &str| {
                fields
                    .get(key)
                    .copied()
                    .with_context(|| format!("KFD property missing: {key}"))
            };
            return Ok(GpuIdentity {
                hip_name,
                pci_bus_id,
                vendor_id: field("vendor_id")?,
                device_id: field("device_id")?,
                gfx_target_version: field("gfx_target_version")?,
                simd_count: field("simd_count")?,
            });
        }
    }
    anyhow::bail!("HIP PCI device {pci_bus_id} absent from KFD topology")
}
fn select_gpu(devices: &[GpuIdentity], requested: Option<usize>) -> Result<usize> {
    let index = if let Some(index) = requested {
        ensure!(index < devices.len(), "GPU index unavailable");
        index
    } else {
        devices
            .iter()
            .position(GpuIdentity::is_9070xt)
            .context("RX 9070 XT not found in HIP/KFD topology")?
    };
    ensure!(
        devices[index].is_9070xt(),
        "selected GPU is {:?}, expected RX 9070 XT gfx1201/128 SIMD",
        devices[index]
    );
    Ok(index)
}
pub(crate) fn device(index: Option<usize>) -> Result<(Device, Value)> {
    let path = std::env::var_os("LIBTORCH").context("LIBTORCH missing; use Nix rocm shell")?;
    let path = Path::new(&path);
    register_hip(path)?;
    ensure!(
        tch::Cuda::is_available(),
        "HIP unavailable; check /dev/kfd and /dev/dri access"
    );
    let devices = (0..tch::Cuda::device_count() as usize)
        .map(|i| identity(path, i))
        .collect::<Result<Vec<_>>>()?;
    let index = select_gpu(&devices, index)?;
    Ok((
        Device::Cuda(index),
        json!({"index":index,"identity":devices[index]}),
    ))
}
fn gpu_tensor(tensor: &Tensor, device: Device) -> Result<()> {
    ensure!(
        tensor.device() == device,
        "GPU placement lost; CPU fallback forbidden"
    );
    Ok(())
}
fn update(
    weight: &Tensor,
    x: &Tensor,
    target: &Tensor,
    optimizer: &mut nn::Optimizer,
    device: Device,
) -> Result<f64> {
    let predicted = x.matmul(weight);
    gpu_tensor(&predicted, device)?;
    let loss = predicted.mse_loss(target, tch::Reduction::Mean);
    gpu_tensor(&loss, device)?;
    let value = loss.double_value(&[]);
    ensure!(value.is_finite(), "nonfinite GPU loss");
    optimizer.zero_grad();
    loss.backward();
    gpu_tensor(&weight.grad(), device)?;
    optimizer.step();
    gpu_tensor(weight, device)?;
    Ok(value)
}
fn host_weights(weight: &Tensor) -> Result<Vec<f32>> {
    Ok(Vec::<f32>::try_from(
        &weight.to_device(Device::Cpu).reshape([-1]),
    )?)
}
fn export(weight: &Tensor, name: &str, out: &Path, repo: &Path) -> Result<f32> {
    let values = host_weights(weight)?;
    let weights = out.join(format!("{name}-weights.json"));
    let decoder = out.join(format!("{name}.onnx"));
    save(&weights, &json!({"size":SIZE,"values":values}))?;
    let status = Command::new("nix")
        .arg("develop")
        .arg(format!("{}#graph", repo.display()))
        .args(["--command", "python"])
        .arg(repo.join("src/graphs/rocm_decoder.py"))
        .arg(&weights)
        .arg(&decoder)
        .stdout(Stdio::from(std::io::stderr()))
        .status()?;
    ensure!(
        status.success(),
        "offline ONNX decoder graph creation failed"
    );
    let mut eye = vec![0f32; (SIZE * SIZE) as usize];
    for i in 0..SIZE as usize {
        eye[i * SIZE as usize + i] = 1.;
    }
    let session = ort::Session::builder()?
        .with_intra_threads(1)?
        .commit_from_file(&decoder)?;
    let input = ort::Tensor::from_array(([SIZE as usize, SIZE as usize], eye))?;
    let output = session.run(ort::inputs![input]?)?;
    let (_, prediction) = output[0].try_extract_raw_tensor::<f32>()?;
    ensure!(
        prediction.len() == values.len(),
        "ONNX decoder shape mismatch"
    );
    let error = prediction
        .iter()
        .zip(values)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    ensure!(
        error <= 1e-5,
        "updated ONNX decoder differs from GPU weights: {error}"
    );
    Ok(error)
}
pub fn gate(out: &Path, repo: &Path, requested: Option<usize>) -> Result<Value> {
    fs::create_dir_all(out)?;
    let start = Instant::now();
    let mut updates = 0;
    let result = (|| -> Result<Value> {
        ensure!(
            Path::new("/dev/kfd").exists(),
            "ROCm device hidden/unavailable; execute outside filesystem sandbox"
        );
        let libtorch =
            std::env::var("LIBTORCH").context("LIBTORCH unset; enter nix develop .#rocm")?;
        register_hip(Path::new(&libtorch))?;
        ensure!(
            tch::Cuda::is_available(),
            "native HIP backend unavailable; CPU fallback forbidden"
        );
        let count = tch::Cuda::device_count() as usize;
        let devices = (0..count)
            .map(|i| identity(Path::new(&libtorch), i))
            .collect::<Result<Vec<_>>>()?;
        let index = select_gpu(&devices, requested)?;
        let identity = &devices[index];
        let device = Device::Cuda(index);
        tch::manual_seed(419);
        let runtime = std::env::var("ORT_DYLIB_PATH")
            .context("ORT_DYLIB_PATH unset; needed to verify updated ONNX export")?;
        ort::init_from(&runtime)
            .with_name("ink-rocm-export-verification")
            .commit()?;
        let store = nn::VarStore::new(device);
        let weight = store
            .root()
            .var("weight", &[SIZE, SIZE], nn::Init::Const(0.));
        let x = Tensor::eye(SIZE, (Kind::Float, device));
        let target = Tensor::ones([SIZE, SIZE], (Kind::Float, device)) * 2.;
        for tensor in [&weight, &x, &target] {
            gpu_tensor(tensor, device)?;
        }
        let mut optimizer = nn::Sgd::default().build(&store, LEARNING_RATE)?;
        let mut losses = Vec::new();
        for _ in 0..10 {
            losses.push(update(&weight, &x, &target, &mut optimizer, device)?);
            updates += 1;
        }
        tch::Cuda::synchronize(index as i64);
        ensure!(losses[9] < losses[0], "GPU loss did not decrease");
        let checkpoint = out.join("checkpoint.safetensors");
        store.save(&checkpoint)?;
        save(
            &out.join("optimizer.json"),
            &json!({"kind":"SGD","learning_rate":LEARNING_RATE,"momentum":0,"weight_decay":0,"updates":updates,"state":"stateless SGD; exact parameters and learning rate suffice"}),
        )?;
        let exported_error = export(&weight, "updated", out, repo)?;
        let mut restored = nn::VarStore::new(device);
        let reloaded = restored
            .root()
            .var("weight", &[SIZE, SIZE], nn::Init::Const(0.));
        restored.load(&checkpoint)?;
        gpu_tensor(&reloaded, device)?;
        ensure!(weight.equal(&reloaded), "GPU checkpoint reload differs");
        let saved: Value = serde_json::from_slice(&fs::read(out.join("optimizer.json"))?)?;
        let mut resumed_optimizer = nn::Sgd::default().build(
            &restored,
            saved["learning_rate"]
                .as_f64()
                .context("checkpoint learning rate")?,
        )?;
        let before = update(&reloaded, &x, &target, &mut resumed_optimizer, device)?;
        updates += 1;
        let after = x
            .matmul(&reloaded)
            .mse_loss(&target, tch::Reduction::Mean)
            .double_value(&[]);
        tch::Cuda::synchronize(index as i64);
        ensure!(
            after < before && !weight.equal(&reloaded),
            "resumed GPU update did not change/improve weights"
        );
        let resumed_error = export(&reloaded, "resumed", out, repo)?;
        let maps = fs::read_to_string("/proc/self/maps")?;
        ensure!(
            !maps.contains("libpython"),
            "Python runtime unexpectedly loaded into Rust trainer"
        );
        Ok(
            json!({"status":"passed","backend":"native-rust-libtorch-rocm","gpu":"AMD Radeon RX 9070 XT","identity":identity,"gpu_index":index,"aten_device":format!("{device:?}"),"training_tensor_placement":"HIP GPU; no CPU fallback","weight_updates":updates,"losses":losses,"initial_loss":losses[0],"tenth_preupdate_loss":losses[9],"resumed_loss_before":before,"resumed_loss_after":after,"checkpoint_reload":true,"resume_update":true,"onnx_export_max_error":exported_error,"resumed_onnx_export_max_error":resumed_error,"onnx_inference_backend":"Rust ort CPU","python_runtime_loaded":false,"libtorch":libtorch,"checkpoint_sha256":file_hash(&checkpoint)?,"updated_decoder_sha256":file_hash(&out.join("updated.onnx"))?,"resumed_decoder_sha256":file_hash(&out.join("resumed.onnx"))?,"fixture":"128x128 matrix; proof of backend, not handwriting model","neural_handwriting_complete":false}),
        )
    })();
    let detail = match &result {
        Ok(v) => v.clone(),
        Err(e) => {
            json!({"status":"blocked","backend":"native-rust-libtorch-rocm","weight_updates":updates,"error":format!("{e:#}"),"cpu_fallback":false})
        }
    };
    let report = json!({"elapsed_seconds":start.elapsed().as_secs_f64(),"gate":detail});
    save(&out.join("gate-report.json"), &report)?;
    result.map(|_| report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selects_discrete_gpu_by_topology_not_generic_name_or_ordinal() {
        let igpu = GpuIdentity {
            hip_name: "AMD Radeon Graphics".into(),
            pci_bus_id: "0000:12:00.0".into(),
            vendor_id: 0x1002,
            device_id: 0x164e,
            gfx_target_version: 100306,
            simd_count: 4,
        };
        let discrete = GpuIdentity {
            hip_name: "AMD Radeon Graphics".into(),
            pci_bus_id: "0000:03:00.0".into(),
            vendor_id: 0x1002,
            device_id: 0x7550,
            gfx_target_version: 120001,
            simd_count: 128,
        };
        let devices = [igpu, discrete];
        assert_eq!(select_gpu(&devices, None).unwrap(), 1);
        assert!(select_gpu(&devices, Some(0)).is_err());
        assert_eq!(select_gpu(&devices, Some(1)).unwrap(), 1);
        assert!(select_gpu(&devices[..1], None).is_err());
    }
}
