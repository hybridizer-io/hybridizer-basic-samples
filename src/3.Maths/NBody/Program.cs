using Hybridizer.Runtime.CUDAImports;

namespace NBody
{
    class Program
    {
        static void Main(string[] args)
        {
            // without a window, a few steps of the simulation on the GPU and in C#, compared
            if (args.Contains("--check"))
            {
                Environment.Exit(Check(256 * 16, 10));
            }

            // the window shares its buffers with CUDA (OpenGL interop): CUDA only
            using var window = new RenderingWindow();
            window.Run();
        }

        static int Check(int numBodies, int steps)
        {
            const float deltaT = 0.001F, softeningSquared = 0.00125f, damping = 0.9995f;
            BodyInitializer.Initialize(1.0f, 1.0f, numBodies, out float4[] positions, out float4[] velocities);
            float4[] gpuPos = (float4[])positions.Clone(), gpuVel = (float4[])velocities.Clone(), gpuNewPos = new float4[numBodies];
            float4[] netPos = (float4[])positions.Clone(), netVel = (float4[])velocities.Clone(), netNewPos = new float4[numBodies];

            const int blockDim = 256;
            int gridDim = (numBodies + blockDim - 1) / blockDim;
            Solver.runner.SetDistrib(gridDim, 1, blockDim, 1, 1, blockDim * 16);
            for (int step = 0; step < steps; ++step)
            {
                Solver.wrapped.Solve(gpuNewPos, gpuPos, gpuVel, numBodies, deltaT, softeningSquared, damping, gridDim);
                (gpuPos, gpuNewPos) = (gpuNewPos, gpuPos);
                Reference(netNewPos, netPos, netVel, numBodies, deltaT, softeningSquared, damping);
                (netPos, netNewPos) = (netNewPos, netPos);
            }

            // the GPU may contract products and sums, and its rsqrt is not 1 / sqrt: compare relatively to the cluster
            double maxError = 0.0, scale = 0.0;
            for (int i = 0; i < numBodies; ++i)
            {
                maxError = Math.Max(maxError, Math.Abs(gpuPos[i].x - netPos[i].x) + Math.Abs(gpuPos[i].y - netPos[i].y) + Math.Abs(gpuPos[i].z - netPos[i].z));
                scale = Math.Max(scale, Math.Abs(netPos[i].x) + Math.Abs(netPos[i].y) + Math.Abs(netPos[i].z));
            }
            Console.WriteLine($"{numBodies} bodies, {steps} steps: largest difference of a position {maxError:G6} (positions up to {scale:G6})");
            if (!(maxError <= 1.0E-4 * scale))
            {
                Console.Error.WriteLine("ERROR : the GPU and C# positions differ");
                return 6;
            }
            Console.WriteLine("OK");
            return 0;
        }

        // Solver.Solve in C#, every body in turn, its interactions in the order of the tiles of the GPU
        static void Reference(float4[] newPos, float4[] oldPos, float4[] velocities, int numBodies, float deltaT, float softeningSquared, float damping)
        {
            Parallel.For(0, numBodies, index =>
            {
                var position = oldPos[index];
                var accel = new float3 { x = 0.0F, y = 0.0F, z = 0.0F };
                for (int j = 0; j < numBodies; ++j)
                {
                    accel = Solver.BodyBodyInteraction(softeningSquared, accel, position, oldPos[j]);
                }
                var velocity = velocities[index];
                velocity.x = (velocity.x + accel.x * deltaT) * damping;
                velocity.y = (velocity.y + accel.y * deltaT) * damping;
                velocity.z = (velocity.z + accel.z * deltaT) * damping;
                position.x = position.x + velocity.x * deltaT;
                position.y = position.y + velocity.y * deltaT;
                position.z = position.z + velocity.z * deltaT;
                newPos[index] = position;
                velocities[index] = velocity;
            });
        }
    }
}
