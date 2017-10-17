using System.Drawing;
using Hybridizer.Runtime.CUDAImports;
using System.Diagnostics;
using System.Drawing.Imaging;
using System;

namespace Hybridizer.Basic.Imaging
{
    unsafe class Program
    {
        static void Main(string[] args)
        {
            // open the input image and lock its content for read operations
            Bitmap baseImage = (Bitmap)Image.FromFile("lena512.bmp");
            PixelFormat format = baseImage.PixelFormat;
            var locked = baseImage.LockBits(new Rectangle(0, 0, baseImage.Width, baseImage.Height), ImageLockMode.ReadOnly, format);
            IntPtr imageData = locked.Scan0;
            int imageBytes = baseImage.Width * baseImage.Height;

            HybRunner runner = HybRunner.Cuda("Sobel_Lock_CUDA.dll").SetDistrib(16, 16, 16, 16, 1, 0);

            // allocate device data 
            IntPtr d_input, d_result;
            cuda.Malloc(out d_input, imageBytes);
            cuda.Malloc(out d_result, imageBytes);
            // move image content to the GPU
            cuda.Memcpy(d_input, imageData, imageBytes, cudaMemcpyKind.cudaMemcpyHostToDevice);

            // create a result image with same pixel format (8 bits per pixel) and lock its content for write operations
            Bitmap resImage = new Bitmap(baseImage.Width, baseImage.Height, format);
            BitmapData destData = resImage.LockBits(new Rectangle(0, 0, baseImage.Width, baseImage.Height), ImageLockMode.WriteOnly, format);
            IntPtr dest = destData.Scan0;

            // run the kernel
            runner.Wrap(new Program()).ComputeSobel(d_result, d_input, baseImage.Width, baseImage.Height);
            
            // fetch result back to CPU
            cuda.Memcpy(dest, d_result, imageBytes, cudaMemcpyKind.cudaMemcpyDeviceToHost);
            cuda.DeviceSynchronize();

            // unlock images
            resImage.UnlockBits(destData);
            baseImage.UnlockBits(locked);

            // and save result
            resImage.Palette = baseImage.Palette;
            resImage.Save("lena_sobel.bmp");
            Process.Start("lena_sobel.bmp");
        }

        [EntryPoint]
        public static void ComputeSobel(byte* outputPixel, byte* inputPixel, int width, int height)
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
