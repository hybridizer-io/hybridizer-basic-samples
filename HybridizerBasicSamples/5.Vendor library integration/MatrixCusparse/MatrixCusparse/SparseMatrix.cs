using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MatrixCusparse
{
    public class SparseMatrix
    {
        public float[] data;
        public int[] indices;
        public int[] rows;

        public SparseMatrix() { }

        public SparseMatrix (Dictionary<int, float>[] from)
        {
            rows = new int[from.Length + 1];
            List<int> _indices = new List<int>();
            List<float> _data = new List<float>();

            int colCounter = 0;
            int i = 0;
            for(; i < from.Length; ++i)
            {
                rows[i] = colCounter;
                foreach(var kvp in from[i])
                {
                    _data.Add(kvp.Value);
                    _indices.Add(kvp.Key);
                    colCounter += 1;
                }
            }
            rows[i] = colCounter;

            indices = _indices.ToArray();
            data = _data.ToArray();
        }

        public static SparseMatrix Laplacian_1D(int rowsCount)
        {
            SparseMatrix a = new SparseMatrix();
            a.data = new float[3 * rowsCount - 2];
            a.indices = new int[3 * rowsCount - 2];
            a.rows = new int[rowsCount + 1];

            a.rows[0] = 0;
            a.data[0] = 2.0F;
            a.data[1] = -1.0F;
            a.indices[0] = 0;
            a.indices[1] = 1;
            a.data[a.data.Length - 2] = -1.0F;
            a.data[a.data.Length - 1] = 2.0F;
            a.indices[a.data.Length - 2] = rowsCount - 2;
            a.indices[a.data.Length - 1] = rowsCount - 1;

            int counter = 2;
            int rowIndex = 1;
            for (; rowIndex < rowsCount - 1; ++rowIndex)
            {
                a.rows[rowIndex] = counter;
                a.indices[counter] = rowIndex - 1;
                a.data[counter++] = -1.0F;
                a.indices[counter] = rowIndex;
                a.data[counter++] = 2.0F;
                a.indices[counter] = rowIndex + 1;
                a.data[counter++] = -1.0F;
            }
            a.rows[rowIndex++] = counter;
            a.rows[rowIndex] = counter + 2;

            return a;
        }
    }
}
