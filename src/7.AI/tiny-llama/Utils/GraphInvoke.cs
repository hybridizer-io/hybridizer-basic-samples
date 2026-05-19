using System.Runtime.InteropServices;
using Hybridizer.Runtime.CUDAImports;

namespace LlamaCsharp.Utils;

/// <summary>
/// Direct bindings to the CUDA-graph wrappers shipped in the LlamaCsharp
/// satellite (see <c>intrinsics.cuh</c>). Hybridizer.Runtime.CUDAImports
/// exposes <c>cudaStream*</c>/<c>cudaMemcpy</c> but not the
/// <c>cudaGraph_*</c>/capture APIs we need to collapse the ~378 per-token
/// kernel launches into one <c>cudaGraphLaunch</c>. The satellite has a
/// matching set of <c>extern "C"</c> <c>llama_*</c> entry points we resolve
/// via <see cref="NativeLibrary.GetExport"/> — same pattern as
/// <see cref="CudaInvoke"/> uses for kernel wrappers.
///
/// All entry points return the raw CUDA <c>cudaError_t</c> integer; we wrap
/// most of them in helpers that throw on non-zero.
/// </summary>
public static class GraphInvoke
{
    private delegate int StreamCreateDel(out nint streamOut);
    private delegate int StreamDestroyDel(nint stream);
    private delegate int StreamSynchronizeDel(nint stream);
    private delegate int GraphBeginCaptureDel(nint stream);
    private delegate int GraphEndCaptureDel(nint stream, out nint graphOut);
    private delegate int GraphInstantiateDel(out nint execOut, nint graph);
    private delegate int GraphLaunchDel(nint exec, nint stream);
    private delegate int GraphDestroyDel(nint graph);
    private delegate int GraphExecDestroyDel(nint exec);

    private static StreamCreateDel _streamCreate = null!;
    private static StreamDestroyDel _streamDestroy = null!;
    private static StreamSynchronizeDel _streamSynchronize = null!;
    private static GraphBeginCaptureDel _graphBeginCapture = null!;
    private static GraphEndCaptureDel _graphEndCapture = null!;
    private static GraphInstantiateDel _graphInstantiate = null!;
    private static GraphLaunchDel _graphLaunch = null!;
    private static GraphDestroyDel _graphDestroy = null!;
    private static GraphExecDestroyDel _graphExecDestroy = null!;

    private static bool _resolved;

    private static void EnsureResolved()
    {
        if (_resolved) return;
        nint lib = CudaInvoke.LibraryHandle;
        if (lib == nint.Zero)
            throw new System.InvalidOperationException("GraphInvoke requires CudaInvoke.Initialize() to be called first.");
        _streamCreate       = Resolve<StreamCreateDel>      (lib, "llama_stream_create");
        _streamDestroy      = Resolve<StreamDestroyDel>     (lib, "llama_stream_destroy");
        _streamSynchronize  = Resolve<StreamSynchronizeDel> (lib, "llama_stream_synchronize");
        _graphBeginCapture  = Resolve<GraphBeginCaptureDel> (lib, "llama_graph_begin_capture");
        _graphEndCapture    = Resolve<GraphEndCaptureDel>   (lib, "llama_graph_end_capture");
        _graphInstantiate   = Resolve<GraphInstantiateDel>  (lib, "llama_graph_instantiate");
        _graphLaunch        = Resolve<GraphLaunchDel>       (lib, "llama_graph_launch");
        _graphDestroy       = Resolve<GraphDestroyDel>      (lib, "llama_graph_destroy");
        _graphExecDestroy   = Resolve<GraphExecDestroyDel>  (lib, "llama_graph_exec_destroy");
        _resolved = true;
    }

    private static T Resolve<T>(nint lib, string name) where T : System.Delegate
    {
        nint sym = NativeLibrary.GetExport(lib, name);
        return Marshal.GetDelegateForFunctionPointer<T>(sym);
    }

    private static void Check(int err)
    {
        if (err != 0)
            throw new System.InvalidOperationException($"CUDA error {err} from GraphInvoke call");
    }

    /// <summary>Create a fresh non-default CUDA stream.</summary>
    public static nint CreateStream()
    {
        EnsureResolved();
        Check(_streamCreate(out nint s));
        return s;
    }

    /// <summary>Destroy a stream previously returned by <see cref="CreateStream"/>.</summary>
    public static void DestroyStream(nint stream)
    {
        EnsureResolved();
        Check(_streamDestroy(stream));
    }

    /// <summary>Block the host until all prior work on the stream completes.</summary>
    public static void SynchronizeStream(nint stream)
    {
        EnsureResolved();
        Check(_streamSynchronize(stream));
    }

    /// <summary>Begin graph capture on the stream (Relaxed mode).</summary>
    public static void BeginCapture(nint stream)
    {
        EnsureResolved();
        Check(_graphBeginCapture(stream));
    }

    /// <summary>End graph capture; returns the captured cudaGraph_t handle.</summary>
    public static nint EndCapture(nint stream)
    {
        EnsureResolved();
        Check(_graphEndCapture(stream, out nint graph));
        return graph;
    }

    /// <summary>Instantiate a graph to a re-launchable cudaGraphExec_t.</summary>
    public static nint Instantiate(nint graph)
    {
        EnsureResolved();
        Check(_graphInstantiate(out nint exec, graph));
        return exec;
    }

    /// <summary>Launch an instantiated graph on the stream.</summary>
    public static void Launch(nint exec, nint stream)
    {
        EnsureResolved();
        Check(_graphLaunch(exec, stream));
    }

    /// <summary>Destroy a cudaGraph_t (typically right after instantiate).</summary>
    public static void DestroyGraph(nint graph)
    {
        EnsureResolved();
        Check(_graphDestroy(graph));
    }

    /// <summary>Destroy a cudaGraphExec_t.</summary>
    public static void DestroyExec(nint exec)
    {
        EnsureResolved();
        Check(_graphExecDestroy(exec));
    }
}
