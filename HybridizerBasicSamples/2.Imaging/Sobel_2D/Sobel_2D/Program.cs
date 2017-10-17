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
            Bitmap baseImage = (Bitmap)Image.FromFile("lena512.bmp");
            var locked = baseImage.LockBits(new Rectangle(0, 0, baseImage.Width, baseImage.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            IntPtr imageData = locked.Scan0;
            int imageBytes = baseImage.Width * baseImage.Height * 32;

            HybRunner runner = HybRunner.Cuda("Sobel_Lock.dll").SetDistrib(32, 32, 16, 16, 1, 0);
            IntPtr d_input, d_result;
            cuda.Malloc(out d_input, imageBytes);
            cuda.Malloc(out d_result, imageBytes);
            cuda.Memcpy(d_input, imageData, imageBytes, cudaMemcpyKind.cudaMemcpyDeviceToHost);
            
            Bitmap resImage = new Bitmap(baseImage.Width, baseImage.Height);
            IntPtr dest = resImage.LockBits(new Rectangle(0, 0, baseImage.Width, baseImage.Height), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb).Scan0;
            cuda.Memcpy(dest, d_result, imageBytes, cudaMemcpyKind.cudaMemcpyDeviceToHost);
            cuda.DeviceSynchronize();
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
                        byte topl = inputPixel[(i - 1) * width + j - 1];
                        byte top = inputPixel[i * width + j - 1];
                        byte topr = inputPixel[(i + 1) * width + j - 1];
                        byte l = inputPixel[i * width + j - 1];
                        byte r = inputPixel[i * width + j + 1];
                        byte botl = inputPixel[(i - 1) * width + j + 1];
                        byte bot = inputPixel[i * width + j + 1];
                        byte botr = inputPixel[(i + 1) * width +  j + 1];

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

                        outputPixel[i * width + j] = (byte)output;
                    }
                }
            }
        }
    }
}
