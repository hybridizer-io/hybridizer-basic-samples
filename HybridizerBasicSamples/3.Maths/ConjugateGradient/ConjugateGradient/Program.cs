using System;
using Hybridizer.Basic.Utilities;
using static Hybridizer.Basic.Utilities.VectorOperation;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hybridizer.Basic.Maths
{
    class Program
    {
        static void Main(string[] args)
        {
            int size = 5;
            SparseMatrix A = SparseMatrix.Laplacian_1D(size);
            float[] B = VectorReader.GetRandomVector(size);
            int maxiter = 10;
            float eps = 1.0e-09f;
            float[] X = new float[size];
            Random rand = new Random(2);
            for (int i = 0; i < B.Length; ++i)
            {
                B[i] = 1.0f;
            }

            ConjugateGradient(X, A, B, maxiter, eps);

            for(int i = 0; i < X.Length; ++i)
            {
                Console.WriteLine(X[i]);
            }
        }

        static public void ConjugateGradient(float[] X, SparseMatrix A, float[] B, int maxiter, float eps)
        {
            float[] R = SubstractVector(B, Multiply(A, X, X.Length));
            float[] P = R;
            float[] Xtmp = X;
            float Rtmp;
            float alpha = 0.0f;
            float beta = 0.0f;
            float[] alphaA;
            for (int k = 0; k < maxiter; k++)
            {
                Rtmp = ScalarProductVector(R, R);
                alphaA = Multiply(A, P, A.rows.Length -1);
                alpha = Rtmp / ScalarProductVector(P, alphaA);
                Xtmp = AddVector(Xtmp, MultiplyVectorByFloat(alpha, P));
                R = SubstractVector(R, MultiplyVectorByFloat(alpha, alphaA));
                if (SquaredNormL2(R) < eps*eps)
                {
                    break;
                }
                beta = ScalarProductVector(R, R) / Rtmp;
                P = AddVector(R, MultiplyVectorByFloat(beta, P));
            }

            for (int i = 0; i < X.Length; ++i)
            {
                X[i] = Xtmp[i];
            }
        }

        public static float[] Multiply(SparseMatrix m, float[] v, int N)
        {
            float[] res = new float[m.rows.Length - 1];
            for (int i = 0; i < N; ++i)
            {
                int rowless = m.rows[i];
                int rowup = m.rows[i + 1];
                float tmp = 0.0F;
                for (int j = rowless; j < rowup; ++j)
                {
                    tmp += v[m.indices[j]] * m.data[j];
                }

                res[i] = tmp;
            }

            return res;
        }
        
        public static float ScalarProductVector(float[] A, float[] B)
        {
            float res = 0.0f;
            for (int i = 0; i < A.Length; ++i)
            {
                res += A[i] * B[i];
            }

            return res;
        }
        
        public static float SquaredNormL2(float[] vector)
        {
            float res = 0.0f;
            for (int i = 0; i < vector.Length; ++i)
            {
                res += vector[i] * vector[i];
            }

            return res;

        }
    }
}
