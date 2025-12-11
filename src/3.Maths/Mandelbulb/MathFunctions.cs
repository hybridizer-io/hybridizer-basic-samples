using Hybridizer.Runtime.CUDAImports;
using System;

namespace Mandelbulb
{
    public static class MathFunctions
    {
        public static float dotProduct(float3 v1, float3 v2)
        {
            return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
        }

        [IntrinsicFunction("sincosf")]
        public static void sincosf(float x, out float sinx, out float cosx)
        {
            sinx = (float)Math.Sin(x);
            cosx = (float) Math.Cos(x);
        }

        [IntrinsicFunction("acosf")]
        public static float acosf(float x)
        {
            return (float)Math.Acos(x);
        }

        [IntrinsicFunction("asinf")]
        public static float asinf(float x)
        {
            return (float)Math.Asin(x);
        }

        [IntrinsicFunction("cosf")]
        public static float cosf(float x)
        {
            return (float)Math.Cos(x);
        }

        [IntrinsicFunction("sinf")]
        public static float sinf(float x)
        {
            return (float)Math.Sin(x);
        }

        [IntrinsicFunction("logf")]
        public static float logf(float x)
        {
            return (float)Math.Log(x);
        }

        [IntrinsicFunction("atan2f")]
        public static float atan2f(float x, float y)
        {
            return (float)Math.Atan2(x, y);
        }

        [IntrinsicFunction("powf")]
        public static float powf(float x, float y)
        {
            return (float)Math.Pow(x, y);
        }

        [IntrinsicFunction("sqrtf")]
        public static float sqrtf(float x)
        {
            return (float)Math.Sqrt(x);
        }

        [IntrinsicFunction("rsqrtf")]
        public static float rsqrtf(float x)
        {
            return 1.0F / (float)Math.Sqrt(x);
        }

        public static float3 clampVec(float3 v1, float min, float max)
        {
            v1.x = clamp(v1.x, min, max);
            v1.y = clamp(v1.y, min, max);
            v1.z = clamp(v1.z, min, max);
            return v1;
        }

        const float PI = 3.1415926535897931F;

        public static float toRad(float r)
        {
            return r * PI / 180.0F;
        }

        public static float3 saturate(float3 n)
        {
            return clampVec(n, 0.0F, 1.0F);
        }

        public static float saturate(float n)
        {
            return clamp(n, 0.0F, 1.0F);
        }

        public static float clamp(float n, float m, float M)
        {
            return Math.Max(m, Math.Min(n, M));
        }

        public static float length(float3 z)
        {
            return sqrtf(z.x * z.x + z.y * z.y + z.z * z.z);
        }

        public static void normalize(ref float3 z)
        {
            float invlength = rsqrtf(z.x * z.x + z.y * z.y + z.z * z.z);
            z.x *= invlength;
            z.y *= invlength;
            z.z *= invlength;
        }

        public static void scalarMultiply(ref float3 a, float amount)
        {
            a.x *= amount;
            a.y *= amount;
            a.z *= amount;
        }

        public static void turnOrthogonal(ref float3 v1)
        {
            var inverse = rsqrtf(v1.x * v1.x + v1.z * v1.z);
            var oldX = v1.x;
            v1.x = -inverse * v1.z;
            v1.z = inverse * oldX;
        }

        public static void crossProduct(ref float3 v1, float3 v2)
        {
            var oldX = v1.x;
            var oldY = v1.y;
            v1.x = v2.y * v1.z - v2.z * oldY;
            v1.y = v2.z * oldX - v2.x * v1.z;
            v1.z = v2.x * oldY - v2.y * oldX;
        }
    }
}
