using Hybridizer.Runtime.CUDAImports;

namespace GenericMemoryAccess
{
    [HybridTemplateConcept]
    public interface IMatrix
    {
        [Kernel]
        double this[int x, int y] { get; set; }
        [Kernel]
        int size { get; }
        void Print();
    }

    public class Matrix: IMatrix
    {
        double[] _data;
        int _n;

        [HybridizerIgnore]
        public Matrix(int N)
        {
            _n = N;
            _data = new double[N * N];
        }

        [Kernel]
        public int size => _n;

        [Kernel]
        public double this[int x, int y]
        {
            get
            {
                return _data[x * _n + y];
            }
            set
            {
                _data[x * _n + y] = value;
            }
        }

        [HybridizerIgnore]
        public void Print()
        {
            for(int j = 0; j < _n; j++)
            {
                for(int i = 0; i < _n; ++i)
                {
                    Console.Write("{0}, ", _data[j*_n + i]);
                }
                Console.WriteLine();
            }
        }
    }

    public class Transposed : IMatrix
    {
        double[] _data;
        int _n;

        [HybridizerIgnore]
        public Transposed(int N)
        {
            _n = N;
            _data = new double[N * N];
        }

        [Kernel]
        public int size => _n;

        [Kernel]
        public double this[int x, int y]
        {
            get
            {
                return _data[y * _n + x];
            }
            set
            {
                _data[y * _n + x] = value;
            }
        }

        [HybridizerIgnore]
        public void Print()
        {
            for (int j = 0; j < _n; j++)
            {
                for (int i = 0; i < _n; ++i)
                {
                    Console.Write("{0}, ", _data[j * _n + i]);
                }
                Console.WriteLine();
            }
        }
    }
}
