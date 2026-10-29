{ pkgs }:
pkgs.stdenv.mkDerivation {
  pname = "libtorch-rocm";
  version = "2.11.0-rocm7.2";
  src = pkgs.fetchurl {
    url = "https://download.pytorch.org/whl/rocm7.2/torch-2.11.0%2Brocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl";
    name = "torch-2.11.0+rocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl";
    hash = "sha256-jwKjYhMoirWSHid+E/WVTMYq/nz6iZt51znU7yvmUQs=";
  };
  nativeBuildInputs = [ pkgs.unzip pkgs.autoPatchelfHook ];
  buildInputs = [ pkgs.stdenv.cc.cc.lib pkgs.zlib pkgs.xz pkgs.bzip2 pkgs.zstd ];
  dontUnpack = true;
  # Extract directly into the output: avoid a second 13 GiB staging copy.
  installPhase = ''
    mkdir -p $out
    unzip -q $src 'torch/include/*' 'torch/lib/*' '*dist-info/licenses/*' -d $out
    rm $out/torch/lib/libtorch_python.so
  '';
  preFixup = ''
    addAutoPatchelfSearchPath "$out/torch/lib"
  '';
  meta = { license = pkgs.lib.licenses.bsd3; platforms = [ "x86_64-linux" ]; };
}
