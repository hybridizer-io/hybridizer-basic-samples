using System.Drawing;
using Hybridizer.Runtime.CUDAImports;
using System.Diagnostics;
using System.Drawing.Imaging;
using System;

namespace Hybridizer.Basic.Imaging
{
    class Program
    {
        static void Main(string[] args)
        {
            // open the input image and lock its content for read operations
            Bitmap baseImage = (Bitmap)Image.FromFile("lena512.bmp");
            PixelFormat format = baseImage.PixelFormat;
            var lockedSource = baseImage.LockBits(new Rectangle(0, 0, baseImage.Width, baseImage.Height), ImageLockMode.ReadOnly, format);
            IntPtr srcData = lockedSource.Scan0;
            int imageBytes = baseImage.Width * baseImage.Height;

            // create a result image with same pixel format (8 bits per pixel) and lock its content for write operations
            Bitmap resImage = new Bitmap(baseImage.Width, baseImage.Height, format);
            BitmapData lockedDest = resImage.LockBits(new Rectangle(0, 0, baseImage.Width, baseImage.Height), ImageLockMode.WriteOnly, format);
            IntPtr destData = lockedDest.Scan0;
            
            // pin images memory for cuda
            cuda.HostRegister(srcData, imageBytes, (uint) cudaHostAllocFlags.cudaHostAllocMapped);
            cuda.HostRegister(destData, imageBytes, (uint)cudaHostAllocFlags.cudaHostAllocMapped);
            IntPtr d_input, d_result;
            cuda.HostGetDevicePointer(out d_input, srcData, cudaGetDevicePointerFlags.cudaReserved);
            cuda.HostGetDevicePointer(out d_result, destData, cudaGetDevicePointerFlags.cudaReserved);
            
            // run the kernel
            HybRunner runner = HybRunner.Cuda("Sobel_Lock_CUDA.dll").SetDistrib(32, 32, 16, 16, 1, 0);
            runner.Wrap(new Program()).ComputeSobel(d_result, d_input, baseImage.Width, baseImage.Height);
            cuda.DeviceSynchronize();

            // unregister pinned memory
            cuda.HostUnregister(destData);
            cuda.HostUnregister(srcData);

            // unlock images
            resImage.UnlockBits(lockedDest);
            baseImage.UnlockBits(lockedSource);

            // and save result
            resImage.Palette = baseImage.Palette;
            resImage.Save("lena_sobel.bmp");
            Process.Start("lena_sobel.bmp");
        }

        [EntryPoint]
        public unsafe static void ComputeSobel(byte* outputPixel, byte* inputPixel, int width, int height)
        {
            for (int i = threadIdx.y + blockIdx.y * blockDim.y; i < height; i += blockDim.y * gridDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < width; j += blockDim.x * gridDim.x)
                {
                    int output = 0;
                    if (i > 0 && j > 0 && i < height - 1 && j < width - 1)
                    {
                        int index = i * width + j;
                        byte topl = inputPixel[index - width - 1];
                        byte top = inputPixel[index - width];
                        byte topr = inputPixel[index - width + 1];
                        byte l = inputPixel[index - 1];
                        byte r = inputPixel[index + 1];
                        byte botl = inputPixel[index + width + 1];
                        byte bot = inputPixel[index + 1];
                        byte botr = inputPixel[index + width + 1];

                        output = ((int)(topl + 2 * l + botl - topr - 2 * r - botr) +
                                        (int)(topl + 2 * top + topr - botl - 2 * bot - botr));
                        if (output < 0)
                        {
                            output = -output;
                        }
                        if (output > 255)
                        {
                            output = 255;
                        }
                    }

                    outputPixel[(i * width + j)] = (byte)output;
                }
            }
        }
    }
}
