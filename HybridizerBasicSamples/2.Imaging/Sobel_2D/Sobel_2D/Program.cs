using System.Drawing;
using Hybridizer.Runtime.CUDAImports;
using System.Diagnostics;

namespace Hybridizer.Basic.Imaging
{
    class Program
    {
        static void Main(string[] args)
        {
            Bitmap baseImage = (Bitmap)Image.FromFile("lena512.bmp");
            const int size = 512;

            Bitmap resImage = new Bitmap(size, size);

            byte[,] inputPixels = new byte[size, size];
            byte[,] outputPixels = new byte[size, size];

            ReadImage(inputPixels, baseImage, size);

            HybRunner runner = HybRunner.Cuda("Sobel_2D_CUDA.dll").SetDistrib(32, 32, 16, 16, 1, 0);
            dynamic wrapper = runner.Wrap(new Program());

            wrapper.ComputeSobel(outputPixels, inputPixels, size, 0, size);

            SaveImage("lena-sobel.bmp", outputPixels, size);
            Process.Start("lena-sobel.bmp");
        }

        public static void ReadImage(byte[,] inputPixel, Bitmap image, int size)
        {
            for (int i = 0; i < size; ++i)
            {
                for (int j = 0; j < size; ++j)
                {
                    inputPixel[i, j] = image.GetPixel(i, j).R;
                }
            }
        }

        [EntryPoint]
        public static void ComputeSobel(byte[,] outputPixel, byte[,] inputPixel, int size, int from, int to)
        {
            for (int i = from + threadIdx.y + blockIdx.y * blockDim.y; i < to; i += blockDim.y * gridDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < size; j += blockDim.x * gridDim.x)
                {   
                    int output = 0;
                    if (i > 0 && j > 0 && i < size - 1 && j < size - 1)
                    {
                        byte topl = inputPixel[i-1, j-1];
                        byte top = inputPixel[i, j-1];
                        byte topr = inputPixel[i+1, j-1];
                        byte l = inputPixel[i, j-1];
                        byte r = inputPixel[i, j+1];
                        byte botl = inputPixel[i-1, j+1];
                        byte bot = inputPixel[i, j+1];
                        byte botr = inputPixel[i+1, j+1];

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

                        outputPixel[i, j] = (byte)output;
                    }
                }
            }
        }

        public static void SaveImage(string nameImage, byte[,] outputPixel, int size)
        {
            Bitmap resImage = new Bitmap(size, size);
            int col = 0;
            for (int i = 0; i < size; ++i)
            {
                for (int j = 0; j < size; ++j)
                {
                    col = outputPixel[i, j];
                    resImage.SetPixel(i, j, Color.FromArgb(col, col, col));
                }
            }

            //store the result image.
            resImage.Save(nameImage, System.Drawing.Imaging.ImageFormat.Png);
        }

    }
}
