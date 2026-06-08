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
    in
    {
        devShells.${system}.default = pkgs.mkShell {
            buildInputs = [ pythonTorchEnv ];
        };

        apps.${system} = {
            generate-data = {
                type = "app";
                program = "${pkgs.writeShellScriptBin "generate-data" ''
                    exec ${pythonEnv}/bin/python ${./generate_data.py} "$@"
                ''}/bin/generate-data";
            };

            run-pytorch = {
                type = "app";
                program = "${pkgs.writeShellScriptBin "run-pytorch" ''
                    exec ${pythonTorchEnv}/bin/python ${./run_pytorch.py} "$@"
                ''}/bin/run-pytorch";
            };

            plot-losses = {
                type = "app";
                program = "${pkgs.writeShellScriptBin "plot-losses" ''
                    exec ${pythonPlotEnv}/bin/python ${./plot_losses.py} "$@"
                ''}/bin/plot-losses";
            };

            default = self.apps.${system}.generate-data;
        };
    };
}
