using Hybridizer.Basic.Utilities;
using Hybridizer.Runtime.CUDAImports;
using System;

namespace SimpleMetadataDecorator
{
	class Filters
	{
		public const string A = "filterA";
		public const string B = "filterB";
	}

	class Element
	{
		public double Data;
		public ThreadLocalDictionary<string, IDecorator> Decoration;

		public Element(double _data)
		{
			Data = _data;
			Decoration = new ThreadLocalDictionary<string, IDecorator>(10);
		}
	}

	interface IDecorator
	{
		[Kernel]
		int prop();
	}

	public class A : IDecorator {[Kernel] public int prop() { return 1; } }
	public class B : IDecorator {[Kernel] public int prop() { return 2; } }

	class Program
	{
		[EntryPoint]
		public static void Run(int N, Element[] a, double[] b, string propertyFilter)
		{
			for (int k = threadIdx.x + blockDim.x * blockIdx.x; k < N; k += blockDim.x * gridDim.x)
			{
				IDecorator decoration;
				if (a[k].Decoration.TryGetValue(propertyFilter, out decoration) && decoration is A)
				{
					a[k].Data += b[k];
				}
			}
		}

		static void Main(string[] args)
		{
			cudaDeviceProp prop;
			cuda.GetDeviceProperties(out prop, 0);
			int N = prop.multiProcessorCount * 256;
			var a = new Element[N];
			var a_verif = new Element[N];
			double[] b = new double[N];
			Random random = new Random(42);
			for (int i = 0; i < N; ++i)
			{
				a[i] = new Element((double)i);
				a_verif[i] = new Element((double)i);
				b[i] = 1.0;
				double rand = random.NextDouble();
				if (rand < 0.33)
				{
					a[i].Decoration.Add(Filters.A, new A());
					a_verif[i].Decoration.Add(Filters.A, new A());
				}
				else if (rand < 0.66)
				{
					a[i].Decoration.Add(Filters.B, new A());
					a_verif[i].Decoration.Add(Filters.B, new A());
				}
				else
				{
					a[i].Decoration.Add(Filters.A, new B());
					a_verif[i].Decoration.Add(Filters.A, new B());
				}
			}

			CudaMarshaler.changeAggregation(true);
			var runner = SatelliteLoader.Load();

			Console.WriteLine("Init done");
			dynamic wrap = runner.Wrap(new Program());
			wrap.Run(N, a, b, Filters.A);
			cuda.ERROR_CHECK(cuda.DeviceSynchronize());
			Console.WriteLine("Kernel done");

			Run(N, a_verif, b, Filters.A);

			for (int i = 0; i < N; ++i)
			{
				if (a[i].Data != a_verif[i].Data)
				{
					Console.WriteLine($"ERROR at {i} : {a[i].Data} != {a_verif[i].Data}");
					Environment.Exit(1);
				}
			}

			Console.WriteLine("OK");
		}
	}
}