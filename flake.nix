{
    description = "NNzig - compile-time-configured neural network library in Zig";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
        flake-utils.url = "github:numtide/flake-utils";
    };

    outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
    let
        pkgs = import nixpkgs { inherit system; };

        # ---- Zig build environment ----

        # The Zig compiler.
        zig = pkgs.zig;

        # LLVM's OpenMP backend, selected with `-Dopenmp` (EIGEN_USE_OPENMP).
        openmp = pkgs.llvmPackages.openmp;

        # OpenBLAS backend, selected with `-Dopenblas` (EIGEN_USE_BLAS).
        openblas = pkgs.openblas;

        # Set the environment variables that `zig build` expects.
        buildEnv = ''
            export OPENMP_INCLUDE_PATH="${openmp.dev}/include"
            export OPENMP_LIB_PATH="${openmp}/lib"
            export OPENBLAS_INCLUDE_PATH="${openblas.dev}/include"
            export OPENBLAS_LIB_PATH="${openblas}/lib"
        '';

        # A wrapper for `zig build` that sets the environment variables.
        mkBuildApp = name: step:
        let arg = if step == "" then "" else step; in
        {
            type = "app";
            program = "${pkgs.writeShellScriptBin name ''
                ${buildEnv}
                exec ${zig}/bin/zig build ${arg} "$@"
            ''}/bin/${name}";
        };

        # ---- Python benchmark environments ----

        # Each app gets its own minimal Python environment so `nix run .#app`
        # only pulls in the packages that app needs.
        pythonEnv = pkgs.python314.withPackages (ps: [ ps.numpy ]);
        pythonPlotEnv = pkgs.python314.withPackages (ps: [ ps.numpy ps.matplotlib ]);
        pythonTorchEnv = pkgs.python314.withPackages (ps: [ ps.numpy ps.torch ]);
        pythonEquinoxEnv = pkgs.python314.withPackages (ps: [ ps.numpy ps.equinox ps.optax ]);
        pythonTfEnv = pkgs.python313.withPackages (ps: [ ps.numpy ps.tensorflow ps.keras ]);
        pythonBenchEnv = pkgs.python314;

        # Path to the benchmark scripts.
        benchSrc = "${self}/benchmarks";

        # Build a flake app that runs `script` with `env`. It cds into the
        # benchmarks directory when invoked from the repo root.
        mkPyApp = name: env: script: {
            type = "app";
            program = "${pkgs.writeShellScriptBin name ''
                if [ ! -f config.json ] && [ -d benchmarks ]; then
                    cd benchmarks
                fi
                export PYTHONPATH="${benchSrc}"
                exec ${env}/bin/python ${benchSrc}/${script} "$@"
            ''}/bin/${name}";
        };
    in
    {
        apps = {
            # The same steps `zig build` exposes. `default` mirrors a bare
            # `zig build` (no step = install the benchmark executable).
            test = mkBuildApp "test" "test";
            benchmark = mkBuildApp "benchmark" "benchmark";
            docs = mkBuildApp "docs" "docs";
            default = mkBuildApp "build" "";

            # Python comparison/benchmark apps (previously in benchmarks/flake.nix).
            generate-data = mkPyApp "generate-data" pythonEnv "generate_data.py";
            run-pytorch = mkPyApp "run-pytorch" pythonTorchEnv "runs/run_pytorch.py";
            run-equinox = mkPyApp "run-equinox" pythonEquinoxEnv "runs/run_equinox.py";
            run-tensorflow = mkPyApp "run-tensorflow" pythonTfEnv "runs/run_tensorflow.py";
            plot-losses = mkPyApp "plot-losses" pythonPlotEnv "plot_losses.py";
            plot-resources = mkPyApp "plot-resources" pythonPlotEnv "plot_resources.py";

            # Run the resource benchmark.
            bench-resources = {
                type = "app";
                program = "${pkgs.writeShellScriptBin "bench-resources" ''
                    if [ ! -f benchmarks/bench_resources.py ]; then
                        echo "bench-resources: run from the repository root (benchmarks/ not found in CWD)" >&2
                        exit 1
                    fi
                    export PATH="${pkgs.lib.makeBinPath [ pkgs.time ]}:$PATH"
                    exec ${pythonBenchEnv}/bin/python ./benchmarks/bench_resources.py "$@"
                ''}/bin/bench-resources";
            };
        };

        # A shell where `zig build ...` works directly (env vars preset).
        devShells.default = pkgs.mkShell {
            packages = [ zig openmp openblas ];
            shellHook = buildEnv;
        };
    });
}
