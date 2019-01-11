[Hybridizer Essentials](https://marketplace.visualstudio.com/items?itemName=altimesh.AltimeshHybridizerExtensionEssentials) is a compiler targeting CUDA-enabled GPUS from .Net. Using parallelization patterns, such as Parallel.For, or ditributing parallel work by hand, the user can benefit from the compute power of GPUS without entering the learning curve of CUDA, all within Visual Studio.

### hybridizer-basic-samples
This repo illustrates a few samples for Hybridizer

These samples may be used with Hybridizer Essentials. However, C# code can run with any version of Hybridizer. 
They illustrate features of the solution and are a good starting point for experimenting and developing software based on Hybridizer.

All new code is added to the repo of the latest CUDA version (currently 10.0). Older CUDA versions are still supported, but don't get the new samples. 

## WARNING
CUDA 9/9.1/9.2 and the latest update of visual studio do not work together (v141 toolset).
see <a href="https://devtalk.nvidia.com/default/topic/1027209/cuda-9-0-does-not-work-with-the-latest-vs-2017-update/" target="_blank">devtalk.nvidia.com</a>.
Install the v140 toolset before trying to compile samples with visual 2017, or use CUDA 10.0

## Requirements
Before you start, you first need to check if you have the right environment. 
You need an install of Visual Studio (2012 or later). 
You need a CUDA-enabled GPU and CUDA (8.0 or later) installed (with the CUDA driver). 
Obviously, you need to install <a href="https://marketplace.visualstudio.com/items?itemName=altimesh.AltimeshHybridizerExtensionEssentials" target="_blank">Hybridizer Essentials</a>. 

## Run
Checkout repository, and open Visual Studio. 
Require and validate license from Hybridizer->License Settings Tool window. 
Open HybridizerBasicSamples solution. 
Build solution and run example of your choice. 
After an update, you might need to reload the solution. 

## Example
```csharp
using System;
using System.Linq;
using System.Threading.Tasks;

using Hybridizer.Runtime.CUDAImports;

namespace HybridizerExample
{
    class Program
    {
        [EntryPoint]
        public static void Add(float[] a, float[] b, int N)
        {
            Parallel.For(0, N, i => a[i] += b[i]);
        }

        static void Main(string[] args)
        {
            // Arrange
            const int N = 1024 * 1024 * 32;
            float[] a = Enumerable.Range(0, N).Select(i => (float)i).ToArray();
            float[] b = Enumerable.Range(0, N).Select(i => 1.0F).ToArray();

            // Run
            HybRunner.Cuda().Wrap(new Program()).Add(a, b, N);

            cuda.DeviceSynchronize();


            // Assert
            for(int i = 0; i < N; ++i)
            {
                if(a[i] != (float)i + 1.0F)
                {
                    Console.Error.WriteLine("Error at {0} : {1} != {2}", i, a[i], (float)i + 1.0F);
                    Environment.Exit(6); // abort
                }
            }

            Console.Out.WriteLine("OK");
        }
    }
}
```
```
hybridizer-cuda Program.cs -o a.exe -run
```

> OK


## Documentation
Samples are explained in the [wiki](https://github.com/altimesh/hybridizer-basic-samples/wiki).

You can find API documentation in our [DocFX generated documentation](http://docs.altimesh.com/api/)


## Notes
After building the csproj, you have to build the generated vcxproj manually or put it in your build dependencies using the configuration manager. 
After installing an update, you may need to unload/reload the solution, or even close and restart visual studio. 
