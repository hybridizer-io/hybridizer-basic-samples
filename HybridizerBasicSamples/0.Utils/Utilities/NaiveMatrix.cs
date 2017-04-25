using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hybridizer.Basic.Utilities
{
    public class NaiveMatrix
    {
        private float[] values;

        public int Height { get; private set; }
        public int Width { get; private set; }

        public NaiveMatrix(int width = 1024, int height = 1024)
        {
            this.Height = height;
            this.Width = width;
            this.values = new float[height * width];
        }

        public float this[int i]
        {
            get { return this.values[i]; }

            set { this.values[i] = value; }
        }

        public float[] Values
        {
            get { return this.values; }

            set { this.values = value; }
        }

        public void FillMatrix()
        {
            Random rand = new Random();
            for (int i = 0; i < this.Height; ++i)
            {
                for (int j = 0; j < this.Width; ++j)
                {
                    this[i * this.Width + j] = rand.NextFloat();
                }

            }
        }

        public void WriteMatrix()
        {
            for (int k = 0; k < this.Height; ++k)
            {
                for (int j = 0; j < this.Width; ++j)
                {
                    Console.Write(this[k * this.Width + j].ToString() + " ");
                }
                Console.WriteLine("");
            }
        }

        public override int GetHashCode()
        {
            return values.GetHashCode();
        }

        override public Boolean Equals(Object o)
        {
            if (o == this)
                return true;

            if (o.GetType() != typeof(NaiveMatrix))
                return false;

            NaiveMatrix m = (NaiveMatrix)o;
            if (this.Height != m.Height || this.Width != m.Width)
                return false;

            for (int i = 0; i < this.Height; ++i)
            {
                for (int j = 0; j < this.Width; ++j)
                {
                    if (Math.Abs(this[i * this.Width + j] - m[i * m.Width + j]) > 1.0E-3)
                        return false;
                }
            }

            return true;
        }
    }
}
