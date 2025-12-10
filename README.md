Hybridizer Essentials is a compiler targeting CUDA-enabled GPUS from .Net. Using parallelization patterns, such as Parallel.For, or ditributing parallel work by hand, the user can benefit from the compute power of GPUS without entering the learning curve of CUDA, all within dotnet environment.

### hybridizer-basic-samples
This repo illustrates a few samples for Hybridizer

These samples may be used with Hybridizer Essentials. However, C# code can run with any version of Hybridizer. 
They illustrate features of the solution and are a good starting point for experimenting and developing software based on Hybridizer.

## WARNING
Since takeover, we just support CUDA 13.0
other versions, and plugins will come soon

## Requirements
Before you start, you first need to check if you have the right environment. 
You need a CUDA-enabled GPU and CUDA 13.0 installed (with the CUDA driver). 
You need to install Hybridizer dotnet tool: 
`dotnet tool install -g Hybridizer --version 2.0.2-alpha 

## Run
Checkout repository, and open vscode or terminal
set env variable HYB_JITTER_CUDA_VERSION to 13.0
cd to the sample of your choice 
run dotnet run

## Documentation
Samples are explained in the [wiki](https://github.com/altimesh/hybridizer-basic-samples/wiki).

You can find API documentation in our [DocFX generated documentation](http://docs.altimesh.com/api/)