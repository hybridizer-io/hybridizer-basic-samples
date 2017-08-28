using Hybridizer.Runtime.CUDAImports;
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
            float[] callResult_net = new float[OPT_N];
            float[] putResult_net = new float[OPT_N];
            float[] stockPrice_net = new float[OPT_N];
            float[] optionStrike_net = new float[OPT_N];
            float[] optionYears_net = new float[OPT_N];

            float[] callResult_cuda = new float[OPT_N];
            float[] putResult_cuda = new float[OPT_N];
            
            Random rand = new Random(Guid.NewGuid().GetHashCode());
            for (int i = 0; i < OPT_N; ++i)
            {
                callResult_net[i] = 0.0f;
                callResult_cuda[i] = 0.0f;
                putResult_net[i] = -1.0f;
                putResult_cuda[i] = -1.0f;
                stockPrice_net[i] = (float)rand.NextDouble() * 25.0f + 5.0f;
                optionStrike_net[i] = (float)rand.NextDouble() * 99.0f + 1.0f;
                optionYears_net[i] = (float)rand.NextDouble() * 9.75f + 0.25f;
            }

            HybRunner runner = HybRunner.Cuda("BlackScholes_CUDA.dll").SetDistrib(20, 256);
            dynamic wrapper = runner.Wrap(new Program());
            
            for (int i = 0; i < NUM_ITERATIONS; ++i)
            {
                wrapper.BlackScholes(callResult_cuda,
                             putResult_cuda,
                             stockPrice_net,
                             optionStrike_net,
                             optionYears_net,
                             0, OPT_N);
            }

            for (int i = 0; i < NUM_ITERATIONS; ++i)
            {
                Parallel.For(0, OPT_N, (opt) =>
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

            float maxCallError = 0.0F;
            float maxPutError = 0.0F;
            float callL2Error = 0.0F;
            float putL2Error = 0.0F;
            float callL1Error = 0.0F;
            float putL1Error = 0.0F;
            for (int i = 0; i < OPT_N; ++i)
            {
                float callError = Math.Abs(callResult_net[i] - callResult_cuda[i]);
                float putError = Math.Abs(putResult_net[i] - putResult_cuda[i]);
                callL2Error += callError * callError;
                putL2Error += putError * putError;
                callL1Error += callError;
                putL1Error += putError;
                maxCallError = maxCallError > callError ? maxCallError : callError;
                maxPutError = maxPutError > putError ? maxPutError : putError;
            }

            callL1Error /= (float)OPT_N;
            putL1Error /= (float)OPT_N;
            callL2Error = Sqrtf(callL2Error) / (float)OPT_N;
            putL2Error = Sqrtf(putL2Error) / (float)OPT_N;

            Console.WriteLine("CALL ERRORS : Linf : {0:G17}, L2 : {1:G17}, L1: {2:G17}", maxCallError, callL2Error, callL1Error);
            Console.WriteLine("PUT ERRORS  : Linf : {0:G17}, L2 : {1:G17}, L1: {2:G17}", maxPutError, putL2Error, putL1Error);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining), IntrinsicFunction("fabsf")]
        public static float fabsf(float f)
        {
            return Math.Abs(f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining), IntrinsicFunction("expf")]
        public static float Expf(float f)
        {
            return (float)Math.Exp((double)f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining), IntrinsicFunction("sqrtf")]
        public static float Sqrtf(float f)
        {
            return (float)Math.Sqrt((double)f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining), IntrinsicFunction("logf")]
        public static float Logf(float f)
        {
            return (float)Math.Log((double)f);
        }

        [EntryPoint]
        public static void BlackScholes(
            float[] callResult,
            float[] putResult,
            float[] stockPrice,
            float[] optionStrike,
            float[] optionYears,
            int lineFrom,
            int lineTo
            )
        {
            for (int i = lineFrom + blockDim.x * blockIdx.x + threadIdx.x; i < lineTo; i += blockDim.x * gridDim.x)
            {
                float sqrtT, expRT;
                float f1, f2, CNDF1, CNDF2;

                sqrtT = Sqrtf(optionYears[i]);
                f1 = (Logf(stockPrice[i] / optionStrike[i]) + (RISKFREE + 0.5f * VOLATILITY * VOLATILITY) * optionYears[i]) /
                         (VOLATILITY * sqrtT);
                f2 = f1 - VOLATILITY * sqrtT;

                CNDF1 = CND(f1);
                CNDF2 = CND(f2);

                expRT = Expf(-RISKFREE * optionYears[i]);
                callResult[i] = stockPrice[i] * CNDF1 - optionStrike[i] * expRT * CNDF2;
                putResult[i] = optionStrike[i] * expRT * (1.0f - CNDF2) - stockPrice[i] * (1.0f - CNDF1);

            }
        }

        [Kernel]
        static float CND(float f)
        {
            const float A1 = 0.31938153f;
            const float A2 = -0.356563782f;
            const float A3 = 1.781477937f;
            const float A4 = -1.821255978f;
            const float A5 = 1.330274429f;
            const float RSQRT2PI = 0.39894228040143267793994605993438f;

            float K = 1.0f / (1.0f + 0.2316419f * fabsf(f));

            float cnd = RSQRT2PI * Expf(-0.5f * f * f) *
                        (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

            if (f > 0)
                cnd = 1.0f - cnd;

            return cnd;
        }
    }
}
