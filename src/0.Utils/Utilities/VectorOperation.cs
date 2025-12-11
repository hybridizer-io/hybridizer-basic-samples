using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hybridizer.Basic.Utilities
{
    public class VectorOperation
    {
        public static float[] AddVector(float[] vec1, float[] vec2)
        {
            if(vec1.Length != vec2.Length)
            {
                throw new ArgumentException("the vectors sizes didn't matches!");
            }
            int length = vec1.Length;
            float[] vecRes = new float[length];
            for(int i = 0; i < length; ++i)
            {
                vecRes[i] = vec1[i] + vec2[i];
            }

            return vecRes;
        }

        public static float[] SubstractVector(float[] vec1, float[] vec2)
        {
            if (vec1.Length != vec2.Length)
            {
                throw new ArgumentException("the vectors sizes didn't matches!");
            }
            int length = vec1.Length;
            float[] vecRes = new float[length];
            for (int i = 0; i < length; ++i)
            {
                vecRes[i] = vec1[i] - vec2[i];
            }

            return vecRes;
        }

        public static float[] MultiplyVectorByFloat(float f, float[] vector)
        {
            float[] res = new float[vector.Length];
            for(int i = 0; i < vector.Length; ++i)
            {
                res[i] = f * vector[i];
            }
            return res;
        }
    }
}
