using Hybridizer.Runtime.CUDAImports;
using System;
using System.Runtime.InteropServices;
using Hybridizer.Basic.Utilities;
using Hybridizer.Runtime.CUDAImports.cusparse;

namespace Hybridizer.Basic.Integration
{
    class Program
    {
        static void Main(string[] args)
        {
            const int redo = 30;

            const int rowsCount = 20000000;
            SparseMatrix a = SparseMatrix.Laplacian_1D(rowsCount);
            
            float[] x = VectorReader.GetSplatVector(a.rows.Length - 1, 1.0F);
            float[] b = new float[x.Length];
               
            float alpha = 1.0f;
            float beta = 0.0f;

            cusparseHandle_t handle;
            CUBSPARSE_64_80.cusparseCreate(out handle);

            cusparseOperation_t transA = cusparseOperation_t.CUSPARSE_OPERATION_NON_TRANSPOSE;

            cusparseMatDescr_t descrA;
            CUBSPARSE_64_80.cusparseCreateMatDescr(out descrA);
            CUBSPARSE_64_80.cusparseSetMatType(descrA, cusparseMatrixType_t.CUSPARSE_MATRIX_TYPE_GENERAL);
            CUBSPARSE_64_80.cusparseSetMatIndexBase(descrA , cusparseIndexBase_t.CUSPARSE_INDEX_BASE_ZERO);
            
            for (int i = 0; i < redo; ++i)
            {
               Multiply(handle, transA, a.rows.Length -1, x.Length,a.data.Length,alpha, descrA, a.data,a.rows,a.indices,x,beta,b);
            }
            
            CUBSPARSE_64_80.cusparseDestroy(handle);

            Console.Out.WriteLine("DONE");

        }
        
        public static void Multiply(cusparseHandle_t handle,
                                     cusparseOperation_t transA,
                                     int m,
                                     int n,
                                     int nnz,
                                     float alpha,
                                     cusparseMatDescr_t descrA,
                                     float[] csrValA,
                                     int[] csrRowPtrA,
                                     int[] csrColIndA,
                                     float[] x,
                                     float beta,
                                     float[] b
                                     )
        {
            cusparseScsrmv(handle, transA, m, n, nnz, ref alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, ref beta, b);
        }

        [DllImport("cusparse64_80.dll", EntryPoint = "cusparseScsrmv", CallingConvention = CallingConvention.Cdecl)]
        public static extern cusparseStatus_t cusparseScsrmv(cusparseHandle_t handle,
                                              cusparseOperation_t transA,
                                              int m,
                                              int n,
                                              int nnz,
                                              ref float alpha,
                                              cusparseMatDescr_t descrA,
                                              [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(CudaMarshaler))] float[] csrValA,
                                              [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(CudaMarshaler))] int[] csrRowPtrA,
                                              [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(CudaMarshaler))] int[] csrColIndA,
                                              [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(CudaMarshaler))] float[] x,
                                              ref float beta,
                                              [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(CudaMarshaler))] float[] b);
    }
}
