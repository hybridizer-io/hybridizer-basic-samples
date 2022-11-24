[Hybridizer Essentials](https://marketplace.visualstudio.com/items?itemName=altimesh.AltimeshHybridizerExtensionEssentials) is a compiler targeting CUDA-enabled GPUS from .Net. Using parallelization patterns, such as Parallel.For, or ditributing parallel work by hand, the user can benefit from the compute power of GPUS without entering the learning curve of CUDA, all within Visual Studio.

### hybridizer-basic-samples
This repo illustrates a few samples for Hybridizer

These samples may be used with Hybridizer Essentials. However, C# code can run with any version of Hybridizer. 
They illustrate features of the solution and are a good starting point for experimenting and developing software based on Hybridizer.

## WARNING
We dropped support for 
- CUDA &lt; 10.0
- Visual Studio &lt; 15

## Requirements
Before you start, you first need to check if you have the right environment. 
You need an install of Visual Studio (2015 or later). 
You need a CUDA-enabled GPU and CUDA (10.0 or later) installed (with the CUDA driver). 
Obviously, you need to install <a href="https://marketplace.visualstudio.com/items?itemName=altimesh.AltimeshHybridizerExtensionEssentials" target="_blank">Hybridizer Essentials</a>. 

## Run
Checkout repository, and open Visual Studio. 
Require and validate license from Hybridizer->License Settings Tool window. 
Open HybridizerBasicSamples solution. 
Choose CUDA version in "solution items" > Directory.Build.Props & App.config (11.6 by default)
Build solution and run example of your choice. 
After an update, you might need to reload the solution. 

## Documentation
Samples are explained in the [wiki](https://github.com/altimesh/hybridizer-basic-samples/wiki).

You can find API documentation in our [DocFX generated documentation](http://docs.altimesh.com/api/)


## Notes
After building the csproj, you have to build the generated vcxproj manually or put it in your build dependencies using the configuration manager. 
After installing an update, you may need to unload/reload the solution, or even close and restart visual studio. 
