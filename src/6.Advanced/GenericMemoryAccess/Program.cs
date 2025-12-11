using Hybridizer.Basic.Utilities;
using Hybridizer.Runtime.CUDAImports;


/// this sample has absolutely no mathematical meaning
/// nor performance related features
/// It's just to show how to rely on generics to access memory
/// instead of virtual functions
namespace GenericMemoryAccess
{
    [HybridRegisterTemplate(Specialize =typeof(MatrixOperations<Matrix>))]
    [HybridRegisterTemplate(Specialize = typeof(MatrixOperations<Transposed>))]
    public class MatrixOperations<TMatrix> where TMatrix : IMatrix
    {
        TMatrix _matrix;

        public MatrixOperations(TMatrix matrix)
        {
            _matrix = matrix;
        }

        [Kernel]
        public void IncrementSupDiag() 
        {
            for (int i = threadIdx.x + blockIdx.x * blockDim.x + 1; i < _matrix.size; i += blockDim.x * gridDim.x)
            {
                _matrix[i, i-1] += 1.0;
            }
        }
    }

    internal class Program
    {
        [EntryPoint]
        public static void RunNormal(MatrixOperations<Matrix> m)
        {
            m.IncrementSupDiag();
        }

        [EntryPoint]
        public static void RunTranposed(MatrixOperations<Transposed> m)
        {
            m.IncrementSupDiag();
        }

        static void Main(string[] args)
        {
            var m = new Matrix(4);
            var t = new Transposed(4);
            var opnormal = new MatrixOperations<Matrix>(m);
            var optransposed = new MatrixOperations<Transposed>(t);

            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, 0);

            HybRunner runner = SatelliteLoader.Load().SetDistrib(2, 2, 2, 2, 1, 0);
            dynamic wrapped = runner.Wrap(new Program());

            //IntPtr d_normal = runner.Marshaller.MarshalManagedToNative(opnormal);
            //IntPtr d_t = runner.Marshaller.MarshalManagedToNative(t);
            wrapped.RunNormal(opnormal);
            wrapped.RunTranposed(optransposed);
            cuda.DeviceSynchronize();

            m.Print();
            t.Print();
        }
    }
}