{
    description = "Flake for the python benchmark";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    };

    outputs = { self, nixpkgs, ... } @ inputs:
    let
        system = "x86_64-linux";
        pkgs = import nixpkgs { inherit system; };

        pythonEnv = pkgs.python314.withPackages (ps: [ ps.numpy ]);
        pythonPlotEnv = pkgs.python314.withPackages (ps: [ ps.numpy ps.matplotlib ]);
        pythonTorchEnv = pkgs.python314.withPackages (ps: [ ps.numpy ps.torch ]);

        # ${self} is the store copy of this flake's directory (benchmarks/),
        # which holds the scripts together with their sibling modules
        # (config.py, config.json). Launching the interpreter on a script that
        # lives there puts that directory on sys.path, so `from config import`
        # resolves. All file I/O (config.json/params.zon/dataset/losses) is
        # resolved relative to the working directory, so the wrapper cds into
        # the benchmarks directory first (the store is read-only).
        src = self;

        # Build a flake app that runs `script` with `env`. It cds into the
        # benchmarks directory when invoked from the repo root (no config.json
        # in CWD but ./benchmarks exists), so reads/writes land in the working
        # tree instead of the read-only nix store.
        mkApp = name: env: script: {
            type = "app";
            program = "${pkgs.writeShellScriptBin name ''
                if [ ! -f config.json ] && [ -d benchmarks ]; then
                    cd benchmarks
                fi
                exec ${env}/bin/python ${src}/${script} "$@"
            ''}/bin/${name}";
        };
    in
    {
        devShells.${system}.default = pkgs.mkShell {
            buildInputs = [ pythonTorchEnv ];
        };

        apps.${system} = {
            generate-data = mkApp "generate-data" pythonEnv "generate_data.py";
            run-pytorch = mkApp "run-pytorch" pythonTorchEnv "run_pytorch.py";
            plot-losses = mkApp "plot-losses" pythonPlotEnv "plot_losses.py";
            default = self.apps.${system}.generate-data;
        };
    };
}
