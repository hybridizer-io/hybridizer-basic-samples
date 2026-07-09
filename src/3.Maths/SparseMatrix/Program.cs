using Hybridizer.Runtime.CUDAImports;
using Hybridizer.Basic.Utilities;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace Hybridizer.Basic.Maths
{
    class Program
    {
        static void Main(string[] args)
        {
            TestLaplacianSmall();

            Stopwatch sw = new Stopwatch();
            sw.Start();

            SparseMatrix A = SparseMatrix.Laplacian_1D(10000000);

            float[] X = VectorReader.GetSplatVector(10000000, 1.0F);

            int redo = 2;
            double memoryOperationsSize = redo * (3.0 * (A.data.Length * sizeof(float)) + 2 * A.rows.Length * sizeof(uint) + A.indices.Length * sizeof(uint));
            Console.WriteLine("matrix read --- starting computations");

            float[] B = new float[A.rows.Length - 1];


            sw.Stop();
            Console.WriteLine("Computing time  : {0} ms\n", sw.ElapsedMilliseconds);
            #region CUDA
            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, 0);

            HybRunner runner = SatelliteLoader.Load().SetDistrib(8 * prop.multiProcessorCount, 256);
            dynamic wrapper = runner.Wrap(new Program());

            for (int i = 0; i < redo; ++i)
            {
                wrapper.Multiply(B, A, X, X.Length);
            }
            #endregion
        }
        private static void TestLaplacianSmall()
        {
            uint n = 10;




            Console.WriteLine("=== TEST 1D Laplacian  (n = {0}) ===", n);

            SparseMatrix A_test = SparseMatrix.Laplacian_1D(n);
            PrintSparseMatrix(A_test);
            Console.WriteLine("=== END OF TEST ===\n");
        }
        private static void PrintSparseMatrix(SparseMatrix A)
        {
            Console.WriteLine("Calculated Matrix : ");
            int rowCount = A.rows.Length - 1;
            for (int i = 0; i < rowCount; ++i)
            {
                for (int j = 0; j < rowCount; ++j)
                {
                    Console.Write("{0} ", GetValue(A, i, j));
                }
                Console.WriteLine("|");
            }
        }
        private static float GetValue(SparseMatrix A, int row, int col)
        {
            uint start = A.rows[row];
            uint end = A.rows[row + 1];

            for (uint k = start; k < end; ++k)
            {
                if (A.indices[k] == col)
                {
                    return A.data[k];
                }
            }

            return 0.0F;
        }
        private static void ReadArguments(string[] args, out string matrixFile, out string? vectorFile)
        {
            if (args.Length < 1)
            {
                throw new ArgumentNullException("no arguments passed ");
            }
            if (!File.Exists(args[0]))
            {
                throw new FileNotFoundException("File doesn't exist");
            }
            if (args.Length >= 2 && File.Exists(args[1]))
            {
                vectorFile = args[1];
            }
            else
            {
                vectorFile = null;
            }
            matrixFile = args[0];

        }

        [EntryPoint]
        public static void Multiply([Out] float[] res, [In] SparseMatrix m, [In] float[] v, int N)
        {
            Parallel.For(0, N, (i) =>
            {
                uint rowless = m.rows[i];
                uint rowup = m.rows[i + 1];
                float tmp = 0.0F;
                for (uint j = rowless; j < rowup; ++j)
                {
                    tmp += v[m.indices[j]] * m.data[j];
                }
                res[i] = tmp;
            });
        }
    }
}