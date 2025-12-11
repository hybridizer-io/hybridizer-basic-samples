using Hybridizer.Runtime.CUDAImports;
using System;

namespace NBody
{
    public abstract class BodyInitializer
    {
        static float PScale { get; set; }
        static float VScale { get; set; }
        static int NumBodies { get; set; }
        static Random _random = new(42);

        static public void Initialize(float clusterScale,
                                      float velocityScale, int numBodies,
                                      out float4[] positions, out float4[] velocities)
        {
            PScale = clusterScale * Math.Max(1.0f, numBodies / 1024.0f);
            VScale = velocityScale * PScale;
            NumBodies = numBodies;

            positions = new float4[numBodies];
            velocities = new float4[numBodies];
            float4 totalMomentum = new float4();
            for (int i = 0; i < numBodies; ++i)
            {
                if( i < numBodies / 2)
                {
                    positions[i] = float4.make_float4(RandP() + 0.5f * PScale, RandP(), RandP() + 50.0f, RandM());
                    velocities[i] = float4.make_float4(RandV(), RandV() + 0.01f * VScale * positions[i].x * positions[i].x, RandV(), positions[i].w);
                }
                else
                {
                    positions[i] = float4.make_float4(RandP() - 0.5f * PScale, RandP(), RandP() + 50.0f, RandM());
                    velocities[i]  = float4.make_float4(RandV(), RandV() - 0.01f * VScale * positions[i].x * positions[i].x, RandV(), positions[i].w);
                }
                totalMomentum += Momentum(velocities[i]);
            }

            var len = velocities.Length;
            for (int i = 0; i < numBodies; ++i)
            {
                velocities[i].x = velocities[i].x - totalMomentum.x / numBodies / velocities[i].w;
                velocities[i].y = velocities[i].y - totalMomentum.y / numBodies / velocities[i].w;
                velocities[i].z = velocities[i].z - totalMomentum.z / numBodies / velocities[i].w;
            }
        }

        static float4 Momentum(float4 velocity)
        {
            // we store mass in velocity.w
            var mass = velocity.w;
            return float4.make_float4(velocity.x * mass,
                              velocity.y * mass,
                              velocity.z * mass,
                              mass);
        }

        static float Rand(float scale, float location)
        {
            return (float)(_random.NextDouble() * scale + location);
        }

        static float RandP()
        {
            return PScale * Rand(1.0f, -0.5f); // [-0.5, 0.5]
        }

        static float RandV()
        {
            return VScale * Rand(1.0f, -0.5f);// [-0.5, 0.5]
        }

        static float RandM()
        {
            return Rand(0.6f, 0.7f); //[0.7, 0.13]
        }
    }
}
