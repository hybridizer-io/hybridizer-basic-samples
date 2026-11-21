#!/usr/bin/env bash
# Builds, transcodes and runs the samples on the GPU, and tells which ones compute what their C# computes.
#
#   HybridizerDir=<build of hybridizer-software-suite> scripts/run-samples.sh [sample ...]
#
# Each sample is built for HybridizerFlavor (METAL on macOS, CUDA elsewhere: src/Directory.Build.props), which
# transcodes it and builds its satellite, then run; it checks its results against C# and exits with another code
# than 0 when they differ. A sample is named by its project (SimplePrintf, not Printf).
#
# HybridizerDir   the runtime and the transcoder built into one directory (README.md)
# OUT             receives the logs and the result (default /tmp/hyb-samples-run): $OUT/matrix.tsv, $OUT/logs/<sample>.*
# TIMEOUT         seconds a sample may run (default 600)
# CONFIGURATION   Debug (default) or Release
#
# matrix.tsv: build (C#), transcode, satellite, msl (the Metal Shading Language compiled by the Metal runtime at the
# first launch: METAL only), run (exit code 0), correct (the check of the sample passed; "-" for a sample that checks
# nothing), the time of the run in seconds, and the times the sample measures itself (GPU, C#), in milliseconds.
set -uo pipefail
unset Platform PLATFORM RuntimeIdentifier 2>/dev/null || true
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../src" && pwd)"
OUT="${OUT:-/tmp/hyb-samples-run}"
TIMEOUT="${TIMEOUT:-600}"
CONFIGURATION="${CONFIGURATION:-Debug}"
[ -n "${HybridizerDir:-}" ] || { echo "set HybridizerDir to a build of hybridizer-software-suite (README.md)" >&2; exit 1; }
if [ -z "${HybridizerFlavor:-}" ]; then
  if [ "$(uname)" = Darwin ]; then HybridizerFlavor=METAL; else HybridizerFlavor=CUDA; fi
fi
mkdir -p "$OUT/logs"

# sample | project directory (under src) | arguments of the run | what the sample checks
#   check: "self" when it compares to C# and fails otherwise, "lines:<text>:<n>" when its output must have n lines
#   holding text (what the GPU prints), "none" when it only runs
SAMPLES="
Builtin|1.Simple/Builtin||self
HelloWorld|1.Simple/HelloWorld||self
InOut|1.Simple/InOut||self
Intrinsics|1.Simple/Intrinsics||self
Malloc|1.Simple/Malloc||self
Mandelbrot|1.Simple/Mandelbrot||self
SimplePrintf|1.Simple/Printf||lines:hello from thread:13
Recursion|1.Simple/Recursion||self
Reduction|1.Simple/Reduction||self
Sobel|2.Imaging/Sobel||self
Sobel_2D|2.Imaging/Sobel_2D||self
ConjugateGradient|3.Maths/ConjugateGradient||self
MonteCarloHeatEquation|3.Maths/MonteCarloHeatEquation||self
NBody|3.Maths/NBody|--check|self
NaiveMatrix|3.Maths/NaiveMatrix||self
Newton|3.Maths/Newton||self
SharedMatrix|3.Maths/SharedMatrix||self
SparseMatrix|3.Maths/SparseMatrix||self
BlackScholes|4.Finance/BlackScholes||self
BlackScholesFloat4|4.Finance/BlackScholesFloat4||self
StrategyBacktest|4.Finance/StrategyBacktest|--synthetic --no-sequential|self
ConstantMemory|5.CUDA_runtime/ConstantMemory||self
Streams|5.CUDA_runtime/Streams||self
GenericFunctions|6.Advanced/GenericFunctions||self
GenericMemoryAccess|6.Advanced/GenericMemoryAccess||self
GenericReduction|6.Advanced/GenericReduction||self
InterfacesReduction|6.Advanced/InterfacesReduction||self
LambdaReduction|6.Advanced/LambdaReduction||self
SimpleMetadataDecorator|6.Advanced/SimpleMetadataDecorator||self
"

declare -a ONLY=("$@")
selected() {
  [ ${#ONLY[@]} -eq 0 ] && return 0
  local s; for s in "${ONLY[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1
}

# the times a sample prints about itself: "<gpu ms>|<c# ms>"
timings() {
  local name="$1" log="$2" gpu="" net=""
  case "$name" in
    InOut) gpu="$(sed -n 's/^in\/out attributes time : \([0-9]*\) ms.*/\1/p' "$log")" ;;
    Mandelbrot) gpu="$(sed -n 's/.*(GPU) : \([0-9]*\) ms.*/\1/p' "$log")"; net="$(sed -n 's/.*(C#) : \([0-9]*\) ms.*/\1/p' "$log")" ;;
    Newton) gpu="$(sed -n 's/^GPU time per image : \([0-9]*\) ms.*/\1/p' "$log")"; net="$(sed -n 's/^C# time per image : \([0-9]*\) ms.*/\1/p' "$log")" ;;
    BlackScholes) gpu="$(sed -n 's/^GPU time per iteration : \([0-9.]*\) ms.*/\1/p' "$log")"; net="$(sed -n 's/^C# time ([0-9]* cores) : \([0-9.]*\) ms.*/\1/p' "$log")" ;;
    StrategyBacktest) gpu="$(sed -n 's/^Running GPU native kernel\.\.\. *\([0-9]*\) ms.*/\1/p' "$log")"
                      net="$(sed -n 's/^Running CPU parallel.*\.\.\. *\([0-9]*\) ms.*/\1/p' "$log")" ;;
  esac
  echo "${gpu:--}|${net:--}"
}

RESULT="$OUT/matrix.tsv"
printf "sample\tbuild\ttranscode\tsatellite\tmsl\trun\tcorrect\tseconds\tgpu_ms\tcsharp_ms\tfirst error\n" > "$RESULT"
printf "%-24s %-6s %-9s %-9s %-5s %-5s %-7s %8s %7s %9s  %s\n" sample build transcode satellite msl run correct seconds gpu_ms csharp_ms "first error"
echo "$SAMPLES" | while IFS='|' read -r name dir args check; do
  [ -n "$name" ] || continue
  selected "$name" || continue
  log="$OUT/logs/$name"
  rm -f "$log".*
  build=- transcode=- satellite=- msl=- run=- correct=- seconds=- first=""
  project="$SRC/$dir"
  if dotnet build "$project" -c "$CONFIGURATION" -p:HybridizerDir="$HybridizerDir" -p:HybridizerFlavor="$HybridizerFlavor" \
       -p:NuGetAudit=false > "$log.build.log" 2>&1; then
    build=ok
  else
    build=FAIL
  fi
  # the build log tells how far the generation went
  if [ "$HybridizerFlavor" = METAL ]; then
    if grep -q '^ *\[DONE\]' "$log.build.log"; then transcode=ok; elif grep -q 'Hybridizer.Application.dll' "$log.build.log"; then transcode=FAIL; fi
    if grep -q "built: .*_METAL.dylib" "$log.build.log"; then satellite=ok; elif [ "$transcode" = ok ]; then satellite=FAIL; fi
  else
    grep -q '^ *\[DONE\]' "$log.build.log" && transcode=ok
    satellite=n/a
  fi
  if [ "$build" = FAIL ]; then
    first="$(grep -m1 -E ' error |: error' "$log.build.log" | sed 's/^[^:]*: *//' | cut -c1-120)"
    [ "$transcode" = ok ] && build=ok   # what failed came after the C# build
  else
    dll="$(sed -n 's/^ *[^ ]* -> \(.*\.dll\)$/\1/p' "$log.build.log" | grep "/$name.dll\$" | tail -1)"
    [ -n "$dll" ] || dll="$(ls "$project"/bin/"$CONFIGURATION"/net*/"$name".dll 2>/dev/null | head -1)"
    start=$(date +%s.%N)
    # shellcheck disable=SC2086
    ( cd "$project" && timeout "$TIMEOUT" dotnet "$dll" $args ) > "$log.run.log" 2>&1
    rc=$?
    end=$(date +%s.%N)
    seconds="$(printf "%.1f" "$(echo "$end - $start" | bc)")"
    if [ $rc -eq 0 ]; then run=ok; elif [ $rc -eq 124 ]; then run=TIMEOUT; else run="FAIL($rc)"; fi
    if [ "$HybridizerFlavor" = METAL ]; then
      # 6: the check of the sample failed, so its kernels ran
      if grep -q 'Metal Shading Language of .* does not compile' "$log.run.log"; then msl=FAIL; elif [ $rc -eq 0 ] || [ $rc -eq 6 ]; then msl=ok; else msl='?'; fi
    else
      msl=n/a
    fi
    case "$check" in
      none) correct=- ;;
      lines:*)
        text="${check#lines:}"; count="${text##*:}"; text="${text%:*}"
        if [ $rc -eq 0 ] && [ "$(grep -c -F "$text" "$log.run.log")" -eq "$count" ]; then correct=ok; else correct=FAIL; fi ;;
      *) if [ $rc -eq 0 ]; then correct=ok; else correct=FAIL; fi ;;
    esac
    if [ $rc -ne 0 ]; then
      first="$(grep -m1 -E ' error: |ERROR|Exception|exception' "$log.run.log" | sed 's/^program_source:[0-9]*:[0-9]*: //' | cut -c1-120)"
    fi
  fi
  t="$(timings "$name" "$OUT/logs/$name.run.log" 2>/dev/null || echo "-|-")"
  gpu="${t%%|*}"; net="${t##*|}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$build" "$transcode" "$satellite" "$msl" "$run" "$correct" "$seconds" "$gpu" "$net" "$first" >> "$RESULT"
  printf "%-24s %-6s %-9s %-9s %-5s %-5s %-7s %8s %7s %9s  %s\n" "$name" "$build" "$transcode" "$satellite" "$msl" "$run" "$correct" "$seconds" "$gpu" "$net" "$first"
done

total="$(awk -F'\t' 'NR>1' "$RESULT" | wc -l | tr -d ' ')"
ran="$(awk -F'\t' 'NR>1 && $6=="ok"' "$RESULT" | wc -l | tr -d ' ')"
correct="$(awk -F'\t' 'NR>1 && $7=="ok"' "$RESULT" | wc -l | tr -d ' ')"
echo "==> $HybridizerFlavor: $ran of $total samples run, $correct of them check their results against C# ($RESULT)"
