### hybridizer-basic-samples
This repo illustrates a few samples for Hybridizer

These samples may be used with Hybridizer Essentials. However, C# code can run with any version of Hybridizer. 
They illustrate features of the solution and are a good starting point for experimenting and developing software based on Hybridizer.

## WARNING
CUDA 9/9.1 and the latest update of visual studio 2017 (15.5.3) do not work together.
see <a href="https://devtalk.nvidia.com/default/topic/1027209/cuda-9-0-does-not-work-with-the-latest-vs-2017-update/" target="_blank">devtalk.nvidia.com</a>.
Don't install latest update if you want CUDA with visual studio 2017. 

## Requirements
Before you start, you first need to check if you have the right environment. 
You need an install of Visual Studio (2012 or later). 
You need a CUDA-enabled GPU and CUDA (8.0 or later) installed (with the CUDA driver). 
Obviously, you need to install <a href="https://marketplace.visualstudio.com/items?itemName=altimesh.AltimeshHybridizerExtensionEssentials" target="_blank">Hybridizer Essentials</a>. 

## Run
Checkout repository, and open Visual Studio. 
Require and validate license from Hybridizer Configuration Tool window (see get_started.mp4 for more details). 
Open HybridizerBasicSamples solution. 
Build solution and run example of your choice. 
After an update, you might need to reload the solution. 

## Wiki
Samples are explained in the [wiki](https://github.com/altimesh/hybridizer-basic-samples/wiki).


## Notes
After building the csproj, you have to build the generated vcxproj manually or put it in your build dependencies using the configuration manager. 
After installing an update, you may need to unload/reload the solution, or even close and restart visual studio. 
