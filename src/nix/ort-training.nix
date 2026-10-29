{ pkgs }:
let
  version = "1.19.2";
  runtime = pkgs.stdenv.mkDerivation {
    pname = "onnxruntime-training-cpu";
    inherit version;
    src = pkgs.fetchurl {
      url = "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.training/1.19.2/microsoft.ml.onnxruntime.training.1.19.2.nupkg";
      hash = "sha256-NH26YiFEmpp/O/z/gsIPs6z+0/PzxUVQiGPDO88wk+E=";
    };
    nativeBuildInputs = [ pkgs.unzip pkgs.autoPatchelfHook ];
    buildInputs = [ pkgs.stdenv.cc.cc.lib ];
    unpackPhase = ''
      unzip -q $src 'runtimes/linux-x64/native/*' 'LICENSE*' 'ThirdPartyNotices.txt' -d source
    '';
    installPhase = ''
      mkdir -p $out/lib $out/share/licenses/onnxruntime-training
      cp source/runtimes/linux-x64/native/*.so $out/lib/
      cp source/LICENSE source/ThirdPartyNotices.txt $out/share/licenses/onnxruntime-training/
    '';
    meta = { license = pkgs.lib.licenses.mit; platforms = [ "x86_64-linux" ]; };
  };
  onnx = pkgs.python312Packages.buildPythonPackage {
    pname = "onnx";
    version = "1.17.0";
    format = "wheel";
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/3d/7c/67f4952d1b56b3f74a154b97d0dd0630d525923b354db117d04823b8b49b/onnx-1.17.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
      hash = "sha256-Tz+1zE4omKxTEqfcA6ZRM90qv5peUg5pr7iApyUeyXo=";
    };
    nativeBuildInputs = [ pkgs.autoPatchelfHook ];
    buildInputs = [ pkgs.stdenv.cc.cc.lib ];
    dependencies = with pkgs.python312Packages; [ numpy protobuf ];
    pythonImportsCheck = [ "onnx" ];
  };
  graphPackage = pkgs.python312Packages.buildPythonPackage {
    pname = "onnxruntime-training-cpu";
    inherit version;
    format = "wheel";
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/8d/f2/3b597c98ed4de0b2408117da779d9904757dc206f8b16aee004bb2a8c8aa/onnxruntime_training_cpu-1.19.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
      hash = "sha256-TI1IBsUHDcbBdeGpaORdlIWI7b/lkLXvnULTUR2A4iA=";
    };
    nativeBuildInputs = [ pkgs.autoPatchelfHook ];
    buildInputs = [ pkgs.stdenv.cc.cc.lib ];
    dependencies = (with pkgs.python312Packages; [ cerberus flatbuffers h5py numpy packaging protobuf sympy setuptools ]) ++ [ onnx ];
    # The upstream training namespace eagerly imports Python optimizers and
    # torch. This package is deliberately graph-only; Rust owns all updates.
    postInstall = ''
      cat > $out/${pkgs.python312.sitePackages}/onnxruntime/training/__init__.py <<'PYTHON'
# Microsoft ONNX Runtime, MIT license. Offline artifact namespace only.
from . import artifacts
__all__ = ["artifacts"]
PYTHON
    '';
    pythonImportsCheck = [ "onnxruntime.training.artifacts" ];
    meta = { license = pkgs.lib.licenses.mit; platforms = [ "x86_64-linux" ]; };
  };
in { inherit runtime graphPackage; }
