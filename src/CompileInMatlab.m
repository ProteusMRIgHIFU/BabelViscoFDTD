GPU_ARCHS={'-gencode=arch=compute_60,code=sm_60', ...
	    '-gencode=arch=compute_61,code=sm_61', ...
        '-gencode=arch=compute_62,code=sm_62', ...
        '-gencode=arch=compute_70,code=sm_70', ...
        '-gencode=arch=compute_75,code=sm_75'};
    
GPU_ARCHS_str='';
for n=1:length(GPU_ARCHS)
    GPU_ARCHS_str=[GPU_ARCHS_str,GPU_ARCHS{n},' '];
end
    

if ismac
    warning('Mac support is outdated, just kept this for reference');
    CUDAToolKitDir='/Developer/NVIDIA/CUDA-9.1';
    CudaSampleDirComon='/Users/spichardo/NVIDIA_CUDA-9.1_Samples/common/inc'
    cmd =[CUDAToolKitDir, '/bin/nvcc --gpu-architecture=compute_61 --gpu-code=sm_61 --resource-usage --cuda -I"', CUDAToolKitDir, '\include" -I"',CudaSampleDirComon,'" -I"', matlabroot ,'/extern/include" -maxrregcount=48 --machine 64  -use_fast_math -DCUDA -DMATLAB_MEX  --output-file "middle_double_cuda2.cpp" "FDTDStaggered3D_with_relaxation_python.cu" '];
elseif isunix
    CUDAToolKitDir='/Developer/NVIDIA/CUDA-9.1';
    CudaSampleDirComon='/Users/spichardo/NVIDIA_CUDA-9.1_Samples/common/inc'
    cmd =[CUDAToolKitDir, '/bin/nvcc ', GPU_ARCHS_str,' --resource-usage --cuda -I"', CUDAToolKitDir, '/include" -I"',CudaSampleDirComon,'" -I"', matlabroot ,'/extern/include" -maxrregcount=48 --machine 64  -use_fast_math -DCUDA -DMATLAB_MEX  --output-file "middle_double_cuda2.cpp" "FDTDStaggered3D_with_relaxation_python.cu" '];
else
    disp('Be sure of specifying the correct location of compilers from your system');
    locationVCL='C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.26.28801\bin\Hostx64\x64';
    CUDAToolKitDir='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2';
    %Please note that the 'C:\ProgramData' is often hidden
    CudaSampleDirComon ='C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.2\common\inc';
    cmd =['"',CUDAToolKitDir,'\bin\nvcc.exe" ',  GPU_ARCHS_str, '--resource-usage --cuda -ccbin "',locationVCL,'"  -I"', CUDAToolKitDir, '\include" -I"',CudaSampleDirComon,'" -I"', matlabroot ,'\extern\include" -maxrregcount=48 --machine 64  -use_fast_math -DCUDA -DMATLAB_MEX -DWIN32 -DWIN64 -DNDEBUG -D_CONSOLE -D_WINDLL -D_MBCS -Xcompiler "/EHsc /W3 /nologo /O2 /Zi /MD"  -o "middle_double_cuda2.cpp" "FDTDStaggered3D_with_relaxation_python.cu"'];

end
disp('***********COMPILING DOUBLE PRECISION CUDA************');

BaseName='FDTDStaggered3D_with_relaxation';

h=system(cmd);
assert(h==0)

if ismac
    cmd ='sed -i '''' ''/typedef __char16_t char16_t/d'' ./middle_double_cuda2.cpp';
    system(cmd);
    cmd ='sed -i '''' ''/typedef __char32_t char32_t/d'' ./middle_double_cuda2.cpp';
    system(cmd);

    cmd =['mex -v  -I"', CUDAToolKitDir, '/include" -I"',CudaSampleDirComon,'"-DMATLAB_MEX -L"', CUDAToolKitDir ,'/lib" -lcudart_static  middle_double_cuda2.cpp -output ',BaseName,'_CUDA_double_mex.mexa64'];

else
    cmd =['mex -v  -I"', CUDAToolKitDir, '\include" -I"',CudaSampleDirComon,'"-DMATLAB_MEX -L"', CUDAToolKitDir ,'/lib/x64" -lcudart_static  middle_double_cuda2.cpp -output ',BaseName,'_CUDA_double_mex.mexa64'];
end
eval(cmd)


delete('middle_double_cuda2.cpp');

disp('***********COMPILING SINGLE PRECISION CUDA************');
if ismac
   cmd =[CUDAToolKitDir, '/bin/nvcc --gpu-architecture=compute_61 --gpu-code=sm_61 --resource-usage --cuda -I"', CUDAToolKitDir, '\include" -I"',CudaSampleDirComon,'" -I"', matlabroot ,'/extern/include" -maxrregcount=48 --machine 64  -use_fast_math -DCUDA -DMATLAB_MEX -DSINGLE_PREC --output-file "middle_single_cuda.cpp" "FDTDStaggered3D_with_relaxation_python.cu" '];

else
    cmd =['"',CUDAToolKitDir,'\bin\nvcc.exe" ',  GPU_ARCHS_str, '--resource-usage --cuda -ccbin "',locationVCL,'"  -I"', CUDAToolKitDir, '\include" -I"',CudaSampleDirComon,'" -I"', matlabroot ,'\extern\include" -maxrregcount=48 --machine 64  -use_fast_math -DCUDA -DMATLAB_MEX -DSINGLE_PREC -DWIN32 -DWIN64 -DNDEBUG -D_CONSOLE -D_WINDLL -D_MBCS -Xcompiler "/EHsc /W3 /nologo /O2 /Zi /MD"  -o "middle_single_cuda.cpp" "FDTDStaggered3D_with_relaxation_python.cu"'];
end

h=system(cmd);
assert(h==0)

if ismac
    cmd ='sed -i '''' ''/typedef __char16_t char16_t/d'' ./middle_single_cuda.cpp';
    system(cmd);
    cmd ='sed -i '''' ''/typedef __char32_t char32_t/d'' ./middle_single_cuda.cpp';
    system(cmd);

    cmd =['mex -v  -I"', CUDAToolKitDir, '/include" -I"',CudaSampleDirComon,'"-DMATLAB_MEX -DSINGLE_PREC -L"', CUDAToolKitDir ,'/lib" -lcudart_static  middle_single_cuda.cpp -output ',BaseName,'_CUDA_single_mex.mexa64'];

else
    cmd =['mex -v -I"', CUDAToolKitDir, '\include" -I"',CudaSampleDirComon,'"-DMATLAB_MEX -DSINGLE_PREC -L"', CUDAToolKitDir ,'/lib/x64" -lcudart_static  middle_single_cuda.cpp -output ',BaseName,'_CUDA_single_mex.mexa64'];

end
eval(cmd)
delete('middle_single_cuda.cpp');

%These are specific for compilers supporting OpenCL (still on Mac) and
%Linux
if ismac || isunix
    disp('***********COMPILING DOUBLE PRECISION OPENCL************');
    if ismac
        cmd =['mex LDFLAGS=''$LDFLAGS -framework OpenCL'' -v  -DMATLAB_MEX -DOPENCL ', BaseName ,'_python.cpp -output ',BaseName,'_OPENCL_double_mex.mexa64'];
    elseif isunix
        cmd =['mex -v  -DMATLAB_MEX -DOPENCL -DWIN32 -DWIN64 -framework OpenCL', BaseName ,'_python.cpp -output ',BaseName,'_OPENCL_double_mex.mexa64'];
    end

    disp(cmd)
    eval(cmd)

    disp('***********COMPILING SINGLE PRECISION OPENCL************');
    if ismac
        cmd =['mex LDFLAGS=''$LDFLAGS -framework OpenCL'' -v  -DMATLAB_MEX -DSINGLE_PREC -DOPENCL ', BaseName ,'_python.cpp -output ',BaseName,'_OPENCL_single_mex.mexa64'];
    else
        cmd =['mex LDFLAGS=''$LDFLAGS -framework OpenCL'' -v  -DMATLAB_MEX -DSINGLE_PREC -DOPENCL -DWIN32 -DWIN64 -framework OpenCL', BaseName ,'_python.cpp -output ',BaseName,'_OPENCL_single_mex.mexa64'];

    end

    disp(cmd)
    eval(cmd)
end


disp('***********COMPILING DOUBLE PRECISION OPENMP************');

if ismac
    cmd =['mex  -v  -DUSE_OPENMP -DMATLAB_MEX  ', BaseName ,'_python.cpp -output ',BaseName,'_double_mex.mexa64'];
else
    cmd =['mex COMPFLAGS=''$COMPFLAGS /openmp'' -v  -DMATLAB_MEX -DWIN32 -DWIN64 ', BaseName ,'_python.cpp -output ',BaseName,'_double_mex.mexa64'];

end
disp(cmd)
eval(cmd)

disp('***********COMPILING single PRECISION OPENMP************');
if ismac
    cmd =['mex  -v -DUSE_OPENMP -DSINGLE_PREC -DMATLAB_MEX  ', BaseName ,'_python.cpp -output ',BaseName,'_single_mex.mexa64'];

else
    cmd =['mex COMPFLAGS=''$COMPFLAGS /openmp'' -v  -DMATLAB_MEX -DSINGLE_PREC -DWIN32 -DWIN64 ', BaseName ,'_python.cpp -output ',BaseName,'_single_mex.mexa64'];

end
disp(cmd)
eval(cmd)
