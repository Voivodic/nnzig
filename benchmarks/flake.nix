{
    description = "Flake for the python benchmark";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    };

    outputs = { self, nixpkgs, ... } @ inputs: 
    let
        # Set the system and the pkgs used
        system = "x86_64-linux";
        pkgs = import nixpkgs { inherit system; };
    in
    {
        # Instructions for the creation of the shell
        devShells.${system}.default = pkgs.mkShell{
            buildInputs = [
                pkgs.python314
                pkgs.python314Packages.numpy
                pkgs.python314Packages.torch
            ];

            shellHook = ''
            '';
        };
    };
}
