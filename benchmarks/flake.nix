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
        pythonEquinoxEnv = pkgs.python314.withPackages (ps: [ ps.numpy ps.equinox ps.optax ]);
        # TensorFlow's prebuilt binary (tensorflow-bin) is not yet available
        # for Python 3.14 in nixpkgs ("unsupported configuration: ..._314"),
        # so the TF benchmark runs on Python 3.13, where the package is fully
        # cached. TF >= 2.16 ships Keras 3 as a SEPARATE package that tf.keras
        # imports lazily, so `keras` must be installed alongside `tensorflow`.
        # The architecture/training/binary-output format are Python-version
        # agnostic, so results stay directly comparable.
        pythonTfEnv = pkgs.python313.withPackages (ps: [ ps.numpy ps.tensorflow ps.keras ]);

        # ${self} is the store copy of this flake's directory (benchmarks/),
        # which holds the scripts together with their sibling modules
        # (config.py, config.json). The run_ scripts live in runs/ (a
        # subdirectory), so PYTHONPATH must include ${src} for
        # `from config import` to resolve — Python only puts the script's own
        # directory (runs/) on sys.path. All file I/O
        # (config.json/params.zon/dataset/losses) is resolved relative to the
        # working directory, so the wrapper cds into the benchmarks directory
        # first (the store is read-only).
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
                export PYTHONPATH="${src}"
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
            run-pytorch = mkApp "run-pytorch" pythonTorchEnv "runs/run_pytorch.py";
            run-equinox = mkApp "run-equinox" pythonEquinoxEnv "runs/run_equinox.py";
            run-tensorflow = mkApp "run-tensorflow" pythonTfEnv "runs/run_tensorflow.py";
            plot-losses = mkApp "plot-losses" pythonPlotEnv "plot_losses.py";
            plot-resources = mkApp "plot-resources" pythonPlotEnv "plot_resources.py";
            default = self.apps.${system}.generate-data;
        };
    };
}
