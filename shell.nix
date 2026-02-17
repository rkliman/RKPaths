{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  # nativeBuildInputs is typically for build tools, 'packages' also works
  packages = with pkgs; [
    python312
    python312Packages.numpy
    python312Packages.scipy
    python312Packages.matplotlib
    python312Packages.shapely
  ];

  shellHook = ''
    echo "Welcome to the classic Nix shell!"
  '';
}