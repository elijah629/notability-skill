{
  description = "One local handwriting and PDF form skill";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/6510408d147d8c5e6d76043e899e9251756f6a75";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      lib = pkgs.lib;
      source = lib.cleanSourceWith {
        src = ./src;
        filter = path: type:
          let relative = lib.removePrefix (toString ./src + "/") (toString path);
          in builtins.elem relative [ "Cargo.toml" "Cargo.lock" "crates" "recipes" "export-flake.nix" ]
            || lib.hasPrefix "crates/" relative || lib.hasPrefix "recipes/" relative;
      };
      # The single distributed skill holds the selected inference weights.
      skill = pkgs.runCommand "notability-skill" { nativeBuildInputs = [ pkgs.unzip ]; } ''
        mkdir -p $out
        unzip -q ${./releases/notability.zip} -d $TMPDIR
        cp -r $TMPDIR/notability-form-latex/. $out/
      '';
      libtorch = import ./src/nix/libtorch-rocm.nix { inherit pkgs; };
      trainingRuntime = import ./src/nix/ort-training.nix { inherit pkgs; };
      makeEngine = native: pkgs.rustPlatform.buildRustPackage ({
        pname = if native then "ink-training-engine" else "ink-engine";
        version = "0.1.0";
        src = source;
        cargoLock.lockFile = ./src/Cargo.lock;
        nativeBuildInputs = [ pkgs.pkg-config pkgs.makeWrapper ];
        buildFeatures = lib.optional native "rocm";
        # Native optimizer fixtures run explicitly, never during an install.
        doCheck = !native;
        meta.mainProgram = "ink";
      } // lib.optionalAttrs native {
        LIBTORCH = "${libtorch}/torch";
        LIBTORCH_CXX11_ABI = "1";
        LD_LIBRARY_PATH = "${libtorch}/torch/lib";
        postInstall = ''
          wrapProgram $out/bin/ink \
            --set ORT_DYLIB_PATH ${trainingRuntime.runtime}/lib/libonnxruntime.so \
            --set LIBTORCH ${libtorch}/torch \
            --prefix LD_LIBRARY_PATH : ${libtorch}/torch/lib
        '';
      });
      engine = makeEngine false;
      nativeEngine = makeEngine true;
      workspace = ''
        repo="''${INK_REPO:-$PWD}"
        if [ ! -f "$repo/src/Cargo.toml" ]; then repo=${self}; fi
        export INK_REPO="$repo"
        work="''${INK_WORKSPACE:-$PWD}"
        if [ -z "''${INK_WORKSPACE:-}" ] && [ -d "$repo/.private/models" ]; then
          work="$repo/.private"
        fi
        export CARGO_TARGET_DIR="''${CARGO_TARGET_DIR:-$work/.cache/cargo}"
      '';
      makeInk = binary: pkgs.writeShellApplication {
        name = "ink";
        text = workspace + ''
          mkdir -p "$work"
          cd "$work"
          export INK_DEFAULT_MODEL="''${INK_DEFAULT_MODEL:-${skill}/assets/models}"
          exec ${binary}/bin/ink "$@"
        '';
      };
      ink = makeInk engine;
      training = makeInk nativeEngine;
      nodeApp = name: entry: pkgs.writeShellApplication {
        inherit name;
        runtimeInputs = [ pkgs.nodejs pkgs.git ];
        text = workspace + ''
          exec node "$repo/${entry}" "$@"
        '';
      };
      install = nodeApp "notability-install" "scripts/install-skill.mjs";
      form = pkgs.writeShellApplication {
        name = "notability-form";
        runtimeInputs = [ pkgs.nodejs ];
        text = ''
          export INK_DECODER="''${INK_DECODER:-${skill}/assets/models/decoder.json}"
          export INK_BINARY="''${INK_BINARY:-${engine}/bin/ink-infer}"
          exec node ${self}/skills/notability-form-latex/scripts/form.mjs "$@"
        '';
      };
      check = nodeApp "notability-check" "scripts/check-public.mjs";
      checkRelease = nodeApp "notability-check-release" "scripts/check-release.mjs";
      test = nodeApp "notability-test" "scripts/test-tools.mjs";
      exportSource = nodeApp "notability-export-source" "scripts/export-public.mjs";
      app = package: name: { type = "app"; program = "${package}/bin/${name}"; };
    in {
      packages.${system} = {
        default = ink;
        inherit ink engine skill training;
        training-runtime = trainingRuntime.runtime;
        graph-tools = trainingRuntime.graphPackage;
      };
      apps.${system} = {
        default = app ink "ink";
        ink = app ink "ink";
        training = app training "ink";
        install = app install "notability-install";
        form = app form "notability-form";
        check = app check "notability-check";
        check-release = app checkRelease "notability-check-release";
        test = app test "notability-test";
        export-source = app exportSource "notability-export-source";
      };
      devShells.${system} = {
        default = pkgs.mkShell {
          packages = [ pkgs.cargo pkgs.rustc pkgs.rustfmt pkgs.clippy pkgs.pkg-config pkgs.nodejs ];
        };
        rocm = pkgs.mkShell {
          packages = [ pkgs.cargo pkgs.rustc pkgs.rustfmt pkgs.clippy pkgs.pkg-config pkgs.nodejs pkgs.stdenv.cc ];
          LIBTORCH = "${libtorch}/torch";
          LIBTORCH_CXX11_ABI = "1";
          LD_LIBRARY_PATH = "${libtorch}/torch/lib";
          ORT_DYLIB_PATH = "${trainingRuntime.runtime}/lib/libonnxruntime.so";
          shellHook = ''
            export CARGO_TARGET_DIR="''${INK_REPO:-$PWD}/src/target/rocm"
          '';
        };
        graph = pkgs.mkShell {
          packages = [ (pkgs.python312.withPackages (p: [ trainingRuntime.graphPackage ])) ];
        };
      };
    };
}
