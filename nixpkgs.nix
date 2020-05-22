let
  # now tracking 20.03-release
  nixpkgsSHA = "10100a97c8964e82b30f180fda41ade8e6f69e41";
   pkgs = import (fetchTarball
   "https://github.com/NixOS/nixpkgs/archive/${nixpkgsSHA}.tar.gz") {
  #pkgs = import <nixos>{
      system = builtins.currentSystem;
      overlays = import ./overlays.nix;
      config = with pkgs.stdenv; {
        whitelistedLicenses = with lib.licenses; [
          unfreeRedistributable
          issl
         ];
        allowUnfreePredicate = pkg: builtins.elem (lib.getName pkg) [
          "cudnn_cudatoolkit"
          "cudatoolkit"
        ];
      };
    };

in pkgs
