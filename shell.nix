let
  pkgs = import ./nixpkgs.nix;
  tfpkgs = import (
    builtins.fetchGit {
      name = "nixos-tensorflow-2";
      url = https://github.com/nixos/nixpkgs;
      ref = "d59b4d07045418bae85a9bdbfdb86d60bc1640bc";}) {};

      tf_packages = with tfpkgs; {
        tf = python37Packages.tensorflowWithCuda;
        tf-tb = python37Packages.tensorflow-tensorboard;
      };

  ml_libs =  pkgs.python37Packages ;
in
  pkgs.mkShell {
    name = "grad";
    buildInputs = with pkgs //  ml_libs; [
      python37
      numpy
      typeguard
      numba
      tensorflow_2
      # tf
      # tf-tb
      matplotlib
      pandas
      dask
      umap-learn
      scipy
      scikitlearn
      h5py
      seaborn
      pytorch
      virtualenv
      torchvision
      R
      rPackages.rmarkdown 
      rPackages.reticulate
   ];
   shellHook = ''
   '';
}
