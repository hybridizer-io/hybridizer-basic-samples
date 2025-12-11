using Hybridizer.Runtime.CUDAImports;
using Hybridizer.Basic;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Formats.Png;
using Hybridizer.Basic.Utilities;

namespace Sobel
{
    class Program
    {
        static void Main(string[] args)
        {
            string path = Path.Combine(AppContext.BaseDirectory, "lena512.bmp");
            var baseImage = Image.Load<Rgba32>(path);
            int height = baseImage.Height, width = baseImage.Width;
            
            var resImage = new Image<Rgba32>(width, height);
            
            byte[] inputPixels = new byte[width * height];
            byte[] outputPixels = new byte[width * height];

            ReadImage(inputPixels, baseImage, width, height);

            HybRunner runner = SatelliteLoader.Load().SetDistrib(32, 32, 16, 16, 1, 0);
            dynamic wrapper = runner.Wrap(new Program());

            wrapper.ComputeSobel(outputPixels, inputPixels, width, height, 0, height);
           

            SaveImage("lena-sobel.bmp", outputPixels, width, height);
        }

        public static void ReadImage(byte[] inputPixel, Image<Rgba32> image, int width, int height)
        {
            for (int i = 0; i < height; ++i)
            {
                for (int j = 0; j < width; ++j)
                {
                    double greyPixel = image[i, j].R * 0.2126 + image[i, j].G * 0.7152 + image[i, j].B * 0.0722;
                    inputPixel[i * height + j] = Convert.ToByte(greyPixel);
                }
            }
        }
        
        [EntryPoint]
        public static void ComputeSobel(byte[] outputPixel, byte[] inputPixel, int width, int height, int from, int to)
        {
            for (int i = from + threadIdx.y + blockIdx.y * blockDim.y; i < to; i += blockDim.y * gridDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < width; j += blockDim.x * gridDim.x)
                {
                    int pixelId = i * width + j;

                    int output = 0;
                    if (i != 0 && j != 0 && i != height - 1 && j != width - 1)
                    {
                        byte topl = inputPixel[pixelId - width - 1];
                        byte top = inputPixel[pixelId - width];
                        byte topr = inputPixel[pixelId - width + 1];
                        byte l = inputPixel[pixelId - 1];
                        byte r = inputPixel[pixelId + 1];
                        byte botl = inputPixel[pixelId + width - 1];
                        byte bot = inputPixel[pixelId + width];
                        byte botr = inputPixel[pixelId + width + 1];

                        int sobelx = topl + (2 * l) + botl - topr - (2 * r) - botr;
                        int sobely = topl + 2 * top + topr - botl - 2 * bot - botr;

                        int squareSobelx = sobelx * sobelx;
                        int squareSobely = sobely * sobely;

                        output = (int)Math.Sqrt(squareSobelx + squareSobely);

                        if (output < 0)
                        {
                            output = -output;
                        }
                        if (output > 255)
                        {
                            output = 255;
                        }

                        outputPixel[pixelId] = (byte)output;
                    }
                }
            }
        }
        
        public static void SaveImage(string nameImage, byte[] outputPixel, int width, int height)
        {
            var resImage = new Image<Rgba32>(width, height);
            byte col = 0;
            for (int i = 0; i < height; ++i)
            {
                for (int j = 0; j < width; ++j)
                {
                    col = outputPixel[i * height + j];
                    resImage[i, j] = Color.FromRgb(col, col, col);
                }
            }

            //store the result image.
            resImage.Save(nameImage, new PngEncoder());
        }
        
    }
}