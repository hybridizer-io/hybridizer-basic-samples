using OpenTK.Graphics.OpenGL;
using System.Runtime.InteropServices;
using Hybridizer.Runtime.CUDAImports;
using OpenTK.Windowing.Desktop;
using OpenTK.Windowing.Common;
using OpenTK.Mathematics;

namespace NBody
{
    public unsafe class RenderingWindow : GameWindow
    {
        private readonly int _numBodies;
        private readonly float _deltaT;
        private readonly float _softeningSquared;
        private readonly float _damping;
        const float _clusterScale = 1.0f;
        const float _velocityScale = 1.0f;

        private readonly uint[] _buffers;
        private readonly IntPtr[] _resources;

        private Random _random;

        private IntPtr _velocities;

        #region constructor
        public RenderingWindow() : base(GameWindowSettings.Default,
                   new NativeWindowSettings()
                   {
                       ClientSize = new OpenTK.Mathematics.Vector2i(800, 600),
                       Title = "Hybridizer N body simulation",
                   })
        {
            Console.WriteLine(sizeof(float4));
            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, 0);
            _random = new Random();
            _numBodies = 256 * 4 * prop.multiProcessorCount;
            _deltaT = 0.001F;
            _softeningSquared = 0.00125f;
            _damping = 0.9995f;
            _buffers = new uint[2];
            _resources = new IntPtr[_buffers.Length];

            initializeBuffers();
            initializeResources();

            cuda.ERROR_CHECK(cuda.Malloc(out _velocities, _numBodies * sizeof(float4)));

            BodyInitializer.Initialize(_clusterScale, _velocityScale, _numBodies, out float4[] hpositions, out float4[] hvelocities);

            // initialize bodies here
            GCHandle posHandle = GCHandle.Alloc(hpositions, GCHandleType.Pinned);
            GCHandle velHandle = GCHandle.Alloc(hvelocities, GCHandleType.Pinned);
            cuda.ERROR_CHECK(cuda.Memcpy(_velocities, velHandle.AddrOfPinnedObject(), _numBodies * sizeof(float4), cudaMemcpyKind.cudaMemcpyHostToDevice));

            MapResources(out nint a, out nint b);
            cuda.ERROR_CHECK(cuda.Memcpy(b, posHandle.AddrOfPinnedObject(), _numBodies * sizeof(float4), cudaMemcpyKind.cudaMemcpyHostToDevice));
            UnMapResources();

            posHandle.Free();
            velHandle.Free();
        }

        private void initializeBuffers()
        {
            for (var i = 0; i < _buffers.Length; i++)
            {
                _buffers[i] = 0;
            }
            GL.GenBuffers(_buffers.Length, _buffers);

            foreach (var buffer in _buffers)
            {
                GL.BindBuffer(BufferTarget.ArrayBuffer, buffer);
                GL.BufferData(BufferTarget.ArrayBuffer, (IntPtr)(Marshal.SizeOf(typeof(float4)) * _numBodies), IntPtr.Zero, BufferUsageHint.DynamicDraw);
                int size;
                unsafe
                {
                    GL.GetBufferParameter(BufferTarget.ArrayBuffer, BufferParameterName.BufferSize, &size);
                }
                if (size != Marshal.SizeOf(typeof(float4)) * _numBodies)
                {
                    throw new Exception("Pixel Buffer Object allocation failed!");
                }

                GL.BindBuffer(BufferTarget.ArrayBuffer, 0);
                cuda.ERROR_CHECK(cuda.GLRegisterBufferObject(buffer));
            }
        }

        private void initializeResources()
        {
            for (var i = 0; i < _buffers.Length; i++)
            {
                var res = IntPtr.Zero;
                cuda.ERROR_CHECK(cuda.GraphicsGLRegisterBuffer(out res, _buffers[i], 0u));
                _resources[i] = res;
            }
        }
        #endregion

        #region random

        class RandomHelper
        {
            private Random _random;
            public float PScale { get; private set; }

            public RandomHelper(float pscale)
            {
                _random = new Random(42);
                PScale = pscale;
            }

            public float Rand(float scale, float location)
            {
                return (float)(_random.NextDouble() * scale + location);
            }

            public float RandP()
            {
                return PScale * Rand(1.0f, -0.5f); 
            }

            public float RandV()
            {
                return PScale * Rand(1.0f, -0.5f);
            }

            public float RandM()
            {
                return Rand(0.6f, 0.7f);
            }
        }

        float4 Momentum(float4 velocity)
        {
            var mass = velocity.w;
            return float4.make_float4(velocity.x * mass,
                              velocity.y * mass,
                              velocity.z * mass,
                              mass);
        }
        #endregion

        #region window management
        protected override void OnLoad()
        {
            VSync = VSyncMode.Off;
            GL.ClearColor(0.0F, 0.0F, 0.0F, 0.0F);
            GL.Enable(EnableCap.DepthTest);
        }

        protected override void OnRenderFrame(FrameEventArgs e)
        {
            base.OnRenderFrame(e);
            SwapPos();
            IntPtr a, b;
            MapResources(out a, out b);

            int blockDim = 256;
            int gridDim = (_numBodies + blockDim - 1) / blockDim;
            Solver.runner.SetDistrib(gridDim, 1, blockDim, 1, 1, blockDim * sizeof(float4));
            Solver.wrapped.Solve(b, a, _velocities, _numBodies, _deltaT, _softeningSquared, _damping, gridDim);
            UnMapResources();

            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            var modelview = Matrix4.LookAt(Vector3.Zero, Vector3.UnitZ, Vector3.UnitY);
            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadMatrix(ref modelview);

            GL.Color3(1.0f, 223.0f / 255.0f, 0.0f); // #FFDF00
            GL.EnableClientState(ArrayCap.VertexArray);
            GL.BindBuffer(BufferTarget.ArrayBuffer, _buffers[1]);
            GL.VertexPointer(4, VertexPointerType.Float, 0, 0);
            GL.DrawArrays(PrimitiveType.Points, 0, _numBodies);
            GL.DisableClientState(ArrayCap.VertexArray);

            GL.Finish();
            SwapBuffers();
        }
        #endregion

        void MapResources (out IntPtr a, out IntPtr b)
        {
            cuda.ERROR_CHECK(cuda.GraphicsResourceSetMapFlags(_resources[0], 1)); // readonly 
            cuda.ERROR_CHECK(cuda.GraphicsResourceSetMapFlags(_resources[1], 2)); // write discard
            cuda.ERROR_CHECK(cuda.GraphicsMapResources(2, _resources, cudaStream_t.NO_STREAM));
            size_t bytesa, bytesb;
            cuda.ERROR_CHECK(cuda.GraphicsResourceGetMappedPointer(out a, out bytesa, _resources[0]));
            cuda.ERROR_CHECK(cuda.GraphicsResourceGetMappedPointer(out b, out bytesb, _resources[1]));
        }

        void UnMapResources()
        {
            cuda.ERROR_CHECK(cuda.GraphicsUnmapResources(2, _resources, cudaStream_t.NO_STREAM));
        }

        void SwapPos()
        {
            (_buffers[1], _buffers[0]) = (_buffers[0], _buffers[1]);
            (_resources[1], _resources[0]) = (_resources[0], _resources[1]);
        }

        #region  disposable
        protected override void Dispose(bool manual)
        {
            foreach (var resource in _resources)
            {
                cuda.ERROR_CHECK(cuda.GraphicsUnregisterResource(resource));
            }
            foreach (var buffer in _buffers)
            {
                cuda.ERROR_CHECK(cuda.GLUnregisterBufferObject(buffer));
            }
            if (_buffers.Length > 0)
            {
                GL.DeleteBuffers(_buffers.Length, _buffers);
            }
            if (manual)
            {
                cuda.Free(_velocities);
            }

            base.Dispose(manual);
        }
        #endregion
    }
}
