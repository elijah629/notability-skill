{
  description = "Personal neural vector inference; no Python or training runtime";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/6510408d147d8c5e6d76043e899e9251756f6a75";
  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      executable = pkgs.runCommand "personal-ink-inference" { nativeBuildInputs = [ pkgs.patchelf ]; } ''
        mkdir -p $out/bin
        cp ${self}/bin/ink-infer $out/bin/ink-infer
        chmod +w $out/bin/ink-infer
        patchelf --set-interpreter ${pkgs.stdenv.cc.bintools.dynamicLinker} --set-rpath ${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.glibc ]} $out/bin/ink-infer
      '';
      launch = pkgs.writeShellScript "personal-ink" ''
        exec ${executable}/bin/ink-infer --decoder ${self}/decoder.json "$@"
      '';
    in { apps.${system}.default = { type = "app"; program = "${launch}"; }; };
}
