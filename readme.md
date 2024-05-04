# CUDA ImGui

CUDA ImGui is a UI for comparing and visualizing image processing CUDA kernels.

The UI is created using the excellent [Dear ImGui](https://github.com/ocornut/imgui). The code is in (`main.cu`). To compile and run, install [Nvidia's CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) then open `cuda-imgui.sln` with Visual Studio.

## Demo
The program runs the selected filter multiple times and plots the average time it takes. The more samples, the more stable the plot will be.

https://github.com/thabetx/cuda-gui/assets/7282243/695d5280-ac8d-498b-9c42-4b4679fe6505
