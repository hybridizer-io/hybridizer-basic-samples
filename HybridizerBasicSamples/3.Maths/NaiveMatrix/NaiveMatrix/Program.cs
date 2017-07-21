using Hybridizer.Runtime.CUDAImports;
using Hybridizer.Basic.Utilities;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hybridizer.Basic.Maths
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                args = new string[] { "512", "512", "512", "512" };
            }
            const int redo = 10;

            int heightA = Convert.ToInt32(args[0]);
            int widthA = Convert.ToInt32(args[1]);
            int heightB = Convert.ToInt32(args[2]);
            int widthB = Convert.ToInt32(args[3]);
            if (widthA != heightB)
            {
                throw new ArgumentException("invalid data -- incompatible matrices");
            }

            Console.WriteLine("Execution Naive matrix mul with sizes ({0}, {1}) x ({2}, {3})", heightA, widthA, heightB, widthB);

            NaiveMatrix matrixA = new NaiveMatrix(widthA, heightA);
            NaiveMatrix matrixB = new NaiveMatrix(widthB, heightB);

            NaiveMatrix res_net = new NaiveMatrix(widthB, heightA);
            NaiveMatrix res_cuda = new NaiveMatrix(widthB, heightA);

            double numberCompute = ((double)matrixA.Height * (double)matrixA.Width * (double)matrixB.Width) * 3.0E-9;
            
            matrixA.FillMatrix();
            matrixB.FillMatrix();

            Random rand = new Random();

            #region CUDA 

            HybRunner runner = HybRunner.Cuda("NaiveMatrix_CUDA.dll").SetDistrib(4, 5, 8, 32, 32, 0);
            dynamic wrapper = runner.Wrap(new Program());
            
            for (int i = 0; i < redo; ++i)
            {
                wrapper.ComputeRowsOfProduct(res_cuda, matrixA, matrixB, 0, res_cuda.Height);
            }
            #endregion

            #region C#

            for (int i = 0; i < redo; ++i)
            {
                Parallel.For(0, res_net.Height, (line) =>
                {
                    ComputeRowsOfProduct(res_net, matrixA, matrixB, line, line + 1);
                });
            }
            #endregion
            
            Console.Out.WriteLine("DONE");
        }

        [EntryPoint]
        public static void ComputeRowsOfProduct(NaiveMatrix resultMatrix, NaiveMatrix matrixA, NaiveMatrix matrixB, int lineFrom, int lineTo)
        {
            int commonSize = matrixA.Width;
            int bWidth = matrixB.Width;
            for (int i = lineFrom + threadIdx.y + blockIdx.y * blockDim.y; i < lineTo; i += blockDim.y * gridDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < bWidth; j += blockDim.x * gridDim.x)
                {
                    resultMatrix[i * bWidth + j] = 0.0f;
                    
                    for (int k = 0; k < commonSize; ++k)
                    {
                        resultMatrix[i * bWidth + j] += (matrixA[i * commonSize + k] * matrixB[k * bWidth + j]);
                    }
                }
            }
        }
    }
}
