using Hybridizer.Runtime.CUDAImports;
using Hybridizer.Basic.Utilities;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Hybridizer.Basic.Finance
{
    class Program
    {
        const int OPT_N = 4000000;
        const int NUM_ITERATIONS = 256;

        const int OPT_SZ = OPT_N * sizeof(float);
        const float RISKFREE = 0.02f;
        const float VOLATILITY = 0.30f;

        static void Main(string[] args)
        {
            float4[] callResult_net = new float4[OPT_N/4];
            float4[] putResult_net = new float4[OPT_N/4];
            float4[] stockPrice_net = new float4[OPT_N/4];
            float4[] optionStrike_net = new float4[OPT_N/4];
            float4[] optionYears_net = new float4[OPT_N/4];

            float4[] callResult_cuda = new float4[OPT_N/4];
            float4[] putResult_cuda = new float4[OPT_N/4];
            
            Random rand = new Random(Guid.NewGuid().GetHashCode());
            for (int i = 0; i < OPT_N/4; ++i)
            {
                callResult_net[i] = new float4(0.0f, 0.0f, 0.0f, 0.0f);
                putResult_net[i] = new float4(-1.0f, -1.0f, -1.0f, -1.0f) ;
                callResult_cuda[i] = new float4(0.0f, 0.0f, 0.0f, 0.0f);
                putResult_cuda[i] = new float4(-1.0f, -1.0f, -1.0f, -1.0f);
                stockPrice_net[i] = rand.NextFloat4(5.0f, 30.0f);
                optionStrike_net[i] = rand.NextFloat4(1.0f, 100.0f);
                optionYears_net[i] = rand.NextFloat4(0.25f, 10f);
            }

            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, 0);
            HybRunner runner = HybRunner.Cuda("BlackScholesFloat4_CUDA.dll").SetDistrib(8 * prop.multiProcessorCount, 256);
            dynamic wrapper = runner.Wrap(new Program());
            
            for (int i = 0; i < NUM_ITERATIONS; ++i)
            {
                wrapper.BlackScholes(callResult_cuda,
                             putResult_cuda,
                             stockPrice_net,
                             optionStrike_net,
                             optionYears_net,
                             0, OPT_N/4);
            }
            for (int i = 0; i < NUM_ITERATIONS; ++i)
            {
                Parallel.For(0, OPT_N/4, (opt) =>
                {
                    BlackScholes(callResult_net,
                                 putResult_net,
                                 stockPrice_net,
                                 optionStrike_net,
                                 optionYears_net,
                                 opt,
                                 opt + 1);
                });
            }

            WriteCalculationError(callResult_net, callResult_cuda, putResult_net, putResult_cuda);

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining), IntrinsicFunction("fabsf")]
        public static float fabsf(float f)
        {
            return Math.Abs(f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining), IntrinsicFunction("__expf")]
        public static float Expf(float f)
        {
            return (float)Math.Exp((double)f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining), IntrinsicFunction("__sqrtf")]
        public static float Sqrtf(float f)
        {
            return (float)Math.Sqrt((double)f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining), IntrinsicFunction("rsqrtf")]
        public static float rsqrtf(float f)
        {
            return 1.0F / (float)Math.Sqrt((double)f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining), IntrinsicFunction("__logf")]
        public static float Logf(float f)
        {
            return (float)Math.Log((double)f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining), IntrinsicFunction("__fdividef")]
        public static float __fdividef(float a, float b)
        {
            return a / b;
        }

        [EntryPoint]
        public static void BlackScholes(
            float4[] callResult,
            float4[] putResult,
            float4[] stockPrice,
            float4[] optionStrike,
            float4[] optionYears,
            int lineFrom,
            int lineTo
            )
        {
            for (int i = lineFrom + blockDim.x * blockIdx.x + threadIdx.x; i < lineTo; i += blockDim.x * gridDim.x)
            {
                float4 sqrtT, expRT, f1, f2, CNDF1, CNDF2;

                float4 years = optionYears[i];
                float4 strike = optionStrike[i];
                float4 price = stockPrice[i];
                float4 call = new float4();
                float4 put = new float4();

                sqrtT.x = __fdividef(1.0F, rsqrtf(years.x));
                sqrtT.y = __fdividef(1.0F, rsqrtf(years.y));
                sqrtT.z = __fdividef(1.0F, rsqrtf(years.z));
                sqrtT.w = __fdividef(1.0F, rsqrtf(years.w));
                f1.x = __fdividef((Logf(__fdividef(price.x, strike.x)) + (RISKFREE + 0.5f * VOLATILITY * VOLATILITY) * years.x), VOLATILITY * sqrtT.x);
                f1.y = __fdividef((Logf(__fdividef(price.y, strike.y)) + (RISKFREE + 0.5f * VOLATILITY * VOLATILITY) * years.y), VOLATILITY * sqrtT.y);
                f1.z = __fdividef((Logf(__fdividef(price.z, strike.z)) + (RISKFREE + 0.5f * VOLATILITY * VOLATILITY) * years.z), VOLATILITY * sqrtT.z);
                f1.w = __fdividef((Logf(__fdividef(price.w, strike.w)) + (RISKFREE + 0.5f * VOLATILITY * VOLATILITY) * years.w), VOLATILITY * sqrtT.w);
                f2.x = f1.x - VOLATILITY * sqrtT.x;
                f2.y = f1.y - VOLATILITY * sqrtT.y;
                f2.z = f1.z - VOLATILITY * sqrtT.z;
                f2.w = f1.w - VOLATILITY * sqrtT.w;

                CNDF1 = CND(f1);
                CNDF2 = CND(f2);

                expRT.x = Expf(-RISKFREE * years.x);
                expRT.y = Expf(-RISKFREE * years.y);
                expRT.z = Expf(-RISKFREE * years.z);
                expRT.w = Expf(-RISKFREE * years.w);
                call.x = price.x * CNDF1.x - strike.x * expRT.x * CNDF2.x;
                call.y = price.y * CNDF1.y - strike.y * expRT.y * CNDF2.y;
                call.z = price.z * CNDF1.z - strike.z * expRT.z * CNDF2.z;
                call.w = price.w * CNDF1.w - strike.w * expRT.w * CNDF2.w;
                put.x = strike.x * expRT.x * (1.0f - CNDF2.x) - price.x * (1.0f - CNDF1.x);
                put.y = strike.y * expRT.y * (1.0f - CNDF2.y) - price.y * (1.0f - CNDF1.y);
                put.z = strike.z * expRT.z * (1.0f - CNDF2.z) - price.z * (1.0f - CNDF1.z);
                put.w = strike.w * expRT.w * (1.0f - CNDF2.w) - price.w * (1.0f - CNDF1.w);

                callResult[i] = call;
                putResult[i] = put;
            }
        }

        [Kernel]
        static float4 CND(float4 f)
        {
            const float A1 = 0.31938153f;
            const float A2 = -0.356563782f;
            const float A3 = 1.781477937f;
            const float A4 = -1.821255978f;
            const float A5 = 1.330274429f;
            const float RSQRT2PI = 0.39894228040143267793994605993438f;

            float4 K, cnd;
            K.x =  __fdividef(1.0F, 1.0f + 0.2316419f * fabsf(f.x));
            K.y =  __fdividef(1.0F, 1.0f + 0.2316419f * fabsf(f.y));
            K.z =  __fdividef(1.0F, 1.0f + 0.2316419f * fabsf(f.z));
            K.w = __fdividef(1.0F, 1.0f + 0.2316419f * fabsf(f.w));

            cnd.x = RSQRT2PI * Expf(-0.5f * f.x * f.x) *
                        (K.x * (A1 + K.x * (A2 + K.x * (A3 + K.x * (A4 + K.x * A5)))));
            cnd.y = RSQRT2PI * Expf(-0.5f * f.y * f.y) *
                        (K.y * (A1 + K.y * (A2 + K.y * (A3 + K.y * (A4 + K.y * A5)))));
            cnd.z = RSQRT2PI * Expf(-0.5f * f.z * f.z) *
                        (K.z * (A1 + K.z * (A2 + K.z * (A3 + K.z * (A4 + K.z * A5)))));
            cnd.w = RSQRT2PI * Expf(-0.5f * f.w * f.w) *
                        (K.w * (A1 + K.w * (A2 + K.w * (A3 + K.w * (A4 + K.w * A5)))));

            if (f.x > 0.0F)
                cnd.x = 1.0f - cnd.x;
            if (f.y > 0.0F)
                cnd.y = 1.0f - cnd.y;
            if (f.z > 0.0F)
                cnd.z = 1.0f - cnd.z;
            if (f.w > 0.0F)
                cnd.w = 1.0f - cnd.w;

            return cnd;
        }

        public static void WriteCalculationError(float4[] callResult1, float4[] callResult2, float4[] putResult1, float4[] putResult2)
        {
            float maxCallError = 0.0F;
            float maxPutError = 0.0F;
            float callL2Error = 0.0F;
            float putL2Error = 0.0F;
            float callL1Error = 0.0F;
            float putL1Error = 0.0F;

            for (int i = 0; i < OPT_N / 4; ++i)
            {
                float4 tmpCallResult = callResult1[i] - callResult2[i];
                float4 tmpPutResult = putResult1[i] - putResult2[i];

                float callErrorX = fabsf(tmpCallResult.x) ;
                float callErrorY = fabsf(tmpCallResult.y) ;
                float callErrorZ = fabsf(tmpCallResult.z) ;
                float callErrorW = fabsf(tmpCallResult.w) ;

                float putErrorX = fabsf(tmpPutResult.x) ;
                float putErrorY = fabsf(tmpPutResult.y) ;
                float putErrorZ = fabsf(tmpPutResult.z) ;
                float putErrorW = fabsf(tmpPutResult.w) ;

                callL2Error += callErrorX * callErrorX + callErrorY * callErrorY + callErrorZ * callErrorZ + callErrorW * callErrorW;

                putL2Error += putErrorX * putErrorX + putErrorY * putErrorY + putErrorZ * putErrorZ + putErrorW * putErrorW;

                callL1Error += callErrorX + callErrorY + callErrorZ + callErrorW;

                putL1Error += putErrorX + putErrorY + putErrorZ + putErrorW;

                maxCallError = maxCallError > callErrorX ? maxCallError : callErrorX;
                maxCallError = maxCallError > callErrorY ? maxCallError : callErrorY;
                maxCallError = maxCallError > callErrorZ ? maxCallError : callErrorZ;
                maxCallError = maxCallError > callErrorW ? maxCallError : callErrorW;

                maxPutError = maxPutError > putErrorX ? maxPutError : putErrorX;
                maxPutError = maxPutError > putErrorY ? maxPutError : putErrorY;
                maxPutError = maxPutError > putErrorZ ? maxPutError : putErrorZ;
                maxPutError = maxPutError > putErrorW ? maxPutError : putErrorW;
            }

            callL1Error /= (float)OPT_N;
            putL1Error /= (float)OPT_N;
            callL2Error = Sqrtf(callL2Error) / (float)OPT_N;
            putL2Error = Sqrtf(putL2Error) / (float)OPT_N;

            Console.WriteLine("CALL ERRORS : Linf : {0:G17}, L2 : {1:G17}, L1: {2:G17}", maxCallError, callL2Error, callL1Error);
            Console.WriteLine("PUT ERRORS  : Linf : {0:G17}, L2 : {1:G17}, L1: {2:G17}", maxPutError, putL2Error, putL1Error);
        }
    }
}
