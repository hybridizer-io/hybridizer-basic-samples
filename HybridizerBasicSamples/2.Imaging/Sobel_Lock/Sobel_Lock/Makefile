build: 
	xbuild /p:Platform=x64 /p:Configuration=Release Sobel_Lock.csproj
	mkdir -p target
	cd target; /opt/altimesh/hybridizer-essentials/bin/hybridize.sh ../hybridizer.base.config
	/usr/local/cuda-8.0/bin/nvcc -O3 -std=c++11 -Xcompiler -fPIC -Xcompiler -shared -m64 target/hybridizer.generated.cpp target/hybridizer.wrappers.cu -L/opt/altimesh/hybridizer-essentials/bin/ -l:libcuda.so -l:libBasicRuntime.so -o target/Sobel_Lock_CUDA.dll

run:
	./run.sh

clean:
	rm -fr target/