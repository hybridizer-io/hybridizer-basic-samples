using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MatrixCusparse
{
    class SparseMatrixReader
    {
        enum MatrixType
        {
            General,
            Symmetric,
            SkewSymmetric,
            Hermitian
        };

        public static Dictionary<int, float>[] ReadMatrixFromFile(string filePath)
        {
            string path = GetPath(filePath);

            int rowCount, colCount;
            string line;
            Dictionary<int, Dictionary<int, float>> tmp = new Dictionary<int, Dictionary<int, float>>();

            StreamReader reader = new StreamReader(path);

            line = reader.ReadLine().ToLowerInvariant();
            MatrixType matrixType = ReadMatrixType(line);
            while ((line = reader.ReadLine()) != null && line.StartsWith("%")) { }

            ReadMatrixSize(line, out rowCount, out colCount);

            while ((line = reader.ReadLine()) != null)
            {
                string[] lineSplitted = line.Split(' ');
                int row, col; float data;
                ReadData(lineSplitted, out row, out col, out data);
                InsertValue(tmp, row, col, data);

                switch (matrixType)
                {
                    case MatrixType.General:
                        break;
                    case MatrixType.Symmetric:
                        InsertValue(tmp, col, row, data);
                        break;
                    case MatrixType.SkewSymmetric:
                        InsertValue(tmp, col, row, -data);
                        break;
                    default:
                        throw new NotImplementedException("only symmetric, general, and skew-symmetric matrices are supported");
                }
            }

            return BuildRawSparseMatrix(rowCount, tmp);
        }

        private static string GetPath(string filePath)
        {
            string path;
            if (Path.IsPathRooted(filePath))
            {
                path = filePath;
            }
            else
            {
                path = Path.Combine(Environment.CurrentDirectory, filePath);
            }

            if (!File.Exists(path))
            {
                throw new FileNotFoundException(path);
            }

            return path;
        }

        private static MatrixType ReadMatrixType(string line)
        {
            if (line.Contains("general"))
            {
                return MatrixType.General;
            }
            else if (line.Contains("symmetric"))
            {
                return MatrixType.Symmetric;
            }
            else if (line.Contains("skew-symmetric"))
            {
                return MatrixType.SkewSymmetric;
            }
            else if (line.Contains("hermitian"))
            {
                return MatrixType.Hermitian;
            }
            else
            {
                throw new ApplicationException("invalid input file -- matrix type not recognized");
            }
        }

        private static void ReadMatrixSize(string line, out int rowCount, out int colCount)
        {
            string[] lineSplitted = line.Split(' ');
            if (lineSplitted.Length != 3)
            {
                throw new ApplicationException("cannot read matrix size");
            }
            else
            {
                if (!int.TryParse(lineSplitted[0], out rowCount))
                {
                    throw new ApplicationException("cannot read matrix size");
                }
                if (!int.TryParse(lineSplitted[1], out colCount))
                {
                    throw new ApplicationException("cannot read matrix size");
                }
            }
        }

        private static void ReadData(string[] lineSplitted, out int row, out int col, out float data)
        {
            if (!Int32.TryParse(lineSplitted[0], out row))
            {
                throw new ApplicationException("invalid line");
            }
            if (!Int32.TryParse(lineSplitted[1], out col))
            {
                throw new ApplicationException("invalid line");
            }
            if (!Single.TryParse(lineSplitted[2], NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out data))
            {
                throw new ApplicationException("invalid line");
            }
        }

        private static void InsertValue(Dictionary<int, Dictionary<int, float>> res, int row, int col, float data)
        {
            if (!res.ContainsKey(row))
            {
                res.Add(row, new Dictionary<int, float>());
                res[row].Add(col, data);
            }
        }

        private static Dictionary<int, float>[] BuildRawSparseMatrix(int rowCount, Dictionary<int, Dictionary<int, float>> tmp)
        {
            Dictionary<int, float>[] res = new Dictionary<int, float>[rowCount];

            for (int i = 0; i < rowCount; ++i)
            {
                if (tmp.ContainsKey(i))
                {
                    res[i] = tmp[i];
                }
                else
                {
                    res[i] = new Dictionary<int, float>();
                }
            }

            return res;
        }
    }
}
