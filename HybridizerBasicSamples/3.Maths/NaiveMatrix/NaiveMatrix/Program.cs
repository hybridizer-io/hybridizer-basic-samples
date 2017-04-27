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
                args = new string[] { "1024", "1024", "1024", "1024" };
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

            Stopwatch watch = new Stopwatch();

            matrixA.FillMatrix();
            matrixB.FillMatrix();

            Random rand = new Random();

            #region CUDA 

            HybRunner runner = HybRunner.Cuda("NaiveMatrix_CUDA.dll").SetDistrib(4, 5, 8, 32, 32, 0);
            dynamic wrapper = runner.Wrap(new Program());

            watch.Start();

            for (int i = 0; i < redo; ++i)
            {
                wrapper.ComputeRowsOfProduct(res_cuda, matrixA, matrixB, 0, res_cuda.Height);
            }

            watch.Stop();

            Console.WriteLine("CUDA GFlop/s : {0}", numberCompute * (double)redo / (watch.ElapsedMilliseconds * 1.0E-03));
            Console.WriteLine("without memcpy         : {0}", numberCompute / (runner.LastKernelDuration.ElapsedMilliseconds * 1.0E-03));
            #endregion

            #region C#
            watch.Restart();

            for (int i = 0; i < redo; ++i)
            {
                Parallel.For(0, res_net.Height, (line) =>
                {
                    ComputeRowsOfProduct(res_net, matrixA, matrixB, line, line + 1);
                });
            }
            watch.Stop();
            Console.WriteLine("C#   GFlop/s :   {0}", numberCompute * (double)redo / (watch.ElapsedMilliseconds * 1.0E-03));
            #endregion

            // verify the results

            if (!res_net.Equals(res_cuda))
                Console.Out.WriteLine("ERROR !");

            //Write the matrix on the console
            //res_cuda.WriteMatrix();

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
