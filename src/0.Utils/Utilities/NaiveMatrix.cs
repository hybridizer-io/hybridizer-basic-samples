namespace Hybridizer.Basic.Utilities
{
    public class NaiveMatrix(int width = 1024, int height = 1024)
    {
        private float[] values = new float[height * width];

        public int Height { get; private set; } = height;
        public int Width { get; private set; } = width;

        public float this[int i]
        {
            get { return values[i]; }

            set { values[i] = value; }
        }

        public float[] Values
        {
            get { return values; }

            set { values = value; }
        }

        public void FillMatrix(float min = 0.0f, float max = 1.0f)
        {
            if(min >max)
            {
                (max, min) = (min, max);
            }

            Random rand = new();
            for (int i = 0; i < Height; ++i)
            {
                for (int j = 0; j < Width; ++j)
                {
                    this[i * Width + j] = rand.NextFloat(min, max);
                }

            }
        }

        public void WriteMatrix()
        {
            for (int k = 0; k < Height; ++k)
            {
                for (int j = 0; j < Width; ++j)
                {
                    Console.Write(this[k * Width + j].ToString() + " ");
                }
                Console.WriteLine("");
            }
        }

        public override int GetHashCode()
        {
            return values.GetHashCode();
        }

        override public bool Equals(object o)
        {
            if (o == this)
                return true;

            if (o.GetType() != typeof(NaiveMatrix))
                return false;

            NaiveMatrix m = (NaiveMatrix)o;
            if (Height != m.Height || Width != m.Width)
                return false;

            for (int i = 0; i < Height; ++i)
            {
                for (int j = 0; j < Width; ++j)
                {
                    if (Math.Abs(this[i * Width + j] - m[i * m.Width + j]) > 1.0E-3)
                        return false;
                }
            }

            return true;
        }
    }
}
