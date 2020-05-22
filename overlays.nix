[
  # top-level pkgs overlays
  (self: super: {
    magma = super.magma.override { mklSupport = true; };

    openmpi = super.openmpi.override { cudaSupport = true; };

    # batteries included :)
    ffmpeg = super.ffmpeg-full.override {
      nonfreeLicensing = true;
      nvenc = true; # nvidia support
    };

    ffmpeg-full = super.ffmpeg-full.override {
      nonfreeLicensing = true;
      nvenc = true; # nvidia support
    };

  })

  # python pkgs overlays
  (self: super: {

    python37Overrides = python37-self: python37-super: {
      numpy = python37-super.numpy.override { blas = super.mkl; };

      pytorch = python37-super.pytorch.override {
        mklSupport = true;
        openMPISupport = true;
        cudaSupport = true;
        buildNamedTensor = true;
        cudaArchList = [
          "5.0"
          "5.2"
          "6.0"
          "6.1"
          "7.0"
          "7.5"
          "7.5+PTX"
        ];
      };

      tensorflow_2 = python37-super.tensorflow_2.override {
        cudaSupport = true;
        cudatoolkit = super.cudatoolkit_10_1;
        cudnn = super.cudnn_cudatoolkit_10_1;
        # https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html
        cudaCapabilities = [
      #    "5.0"
      #    "5.2"
      #    "6.0"
          "6.1"
      #    "7.0"
      #    "7.5"
      #    "10.1"
        ];
        sse42Support = true;
        avx2Support = false;
        fmaSupport = true;

      };
      tensorflow = python37-super.tensorflow.override {
        cudaSupport = true;
        cudatoolkit = super.cudatoolkit_10_1;
        cudnn = super.cudnn_cudatoolkit_10_1;
        # https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html
        cudaCapabilities = [
          "5.0"
          "5.2"
          "6.0"
          "6.1"
          "7.0"
          "7.5"
        ];
        sse42Support = true;
        avx2Support = false;
        fmaSupport = true;

      };

      tensorflow_avx2 = python37-super.tensorflow.override {
        cudaSupport = true;
        cudatoolkit = super.cudatoolkit_10_1;
        cudnn = super.cudnn_cudatoolkit_10_1;
        # https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html
        cudaCapabilities = [
          "5.0"
          "5.2"
          "6.0"
          "6.1"
          "7.0"
          "7.5"
        ];
        sse42Support = true;
        avx2Support = true;
        fmaSupport = true;
      };

      opencv3 = python37-super.opencv3.override {
        enableCuda = true;
        enableFfmpeg = true;
      };

      opencv4 = python37-super.opencv4.override {
        enableCuda = true;
        enableFfmpeg = true;
      };
    };

    python37 =
      super.python37.override { packageOverrides = self.python37Overrides; };

  })
]
