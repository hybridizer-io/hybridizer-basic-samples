using Hybridizer.Basic.Utilities;
using Hybridizer.Runtime.CUDAImports;

namespace NBody
{
    public class Solver
    {
        public static HybRunner runner;
        public static dynamic wrapped;

        static Solver() {
            runner = SatelliteLoader.Load();
            wrapped = runner.Wrap(new Solver());
        }

        [EntryPoint]
        public static void Solve(float4[] newPos, float4[] oldPos, float4[] velocities, int numBodies, float deltaT, float softeningSquared, float damping, int numTiles)
        {
            var index = threadIdx.x + blockIdx.x * blockDim.x;

            if (index >= numBodies)
            {
                return;
            }

            var position = oldPos[index];
            var accel = ComputeBodyAccel(softeningSquared, position, oldPos, numTiles);
            
            // acceleration = force / mass;
            // new velocity = old velocity + acceleration * deltaTime
            // note we factor out the body's mass from the equation, here and in bodyBodyInteraction
            // (because they cancel out).  Thus here force == acceleration
            var velocity = velocities[index];

            velocity.x += accel.x * deltaT;
            velocity.y += accel.y * deltaT;
            velocity.z += accel.z * deltaT;

            velocity.x *= damping;
            velocity.y *= damping;
            velocity.z *= damping;

            // new position = old position + velocity * deltaTime
            position.x = position.x + velocity.x * deltaT;
            position.y = position.y + velocity.y * deltaT;
            position.z = position.z + velocity.z * deltaT;
            
            // store new position and velocity
            newPos[index] = position;
            velocities[index] = velocity;
        }

        [Kernel]
        public static float3 ComputeBodyAccel(float softeningSquared, float4 bodyPos, float4[] positions, int numTiles)
        {
            var sharedPos = new SharedMemoryAllocator<float4>().allocate(blockDim.x);
            var acc = new float3();// 0.0f, 0.0f, 0.0f);
            acc.x = 0.0F; acc.y = 0.0F; acc.z = 0.0F;

            for (var tile = 0; tile < numTiles; tile++)
            {
                sharedPos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];

                CUDAIntrinsics.__syncthreads();

                // This is the "tile_calculation" from the GPUG3 article.
                for (var counter = 0; counter < blockDim.x; counter++)
                {
                    acc = BodyBodyInteraction(softeningSquared, acc, bodyPos, sharedPos[counter]);
                }

                CUDAIntrinsics.__syncthreads();
            }

            return acc;
        }

        [IntrinsicFunction("rsqrtf")]
        public static float rsqrtf(float x)
        {
            return 1.0F / (float) Math.Sqrt(x);
        }

        [Kernel]
        public static float3 BodyBodyInteraction(float softeningSquared, float3 ai, float4 bi, float4 bj)
        {
            float3 r = new()
            {
                x = bj.x - bi.x,
                y = bj.y - bi.y,
                z = bj.z - bi.z
            }; 
            var distSqr = r.x * r.x + r.y * r.y + r.z * r.z + softeningSquared;
            var invDist = rsqrtf(distSqr);
            var invDistCube = invDist * invDist * invDist;
            var s = bj.w * invDistCube;
            float3 res = new()
            {
                x = ai.x + r.x * s,
                y = ai.y + r.y * s,
                z = ai.z + r.z * s
            }; return res;
        }
    }
}
