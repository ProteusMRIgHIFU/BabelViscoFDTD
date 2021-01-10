#include "../mtlpp.hpp"
#include <stdio.h>

int main()
{

    ns::Array<mtlpp::Device>  AllDev= mtlpp::Device::CopyAllDevices();
    for (int n = 0;n<AllDev.GetSize();n++)
    {
      printf("%i %s\n",n,AllDev[n].GetName().GetCStr());
    }

    mtlpp::Device device = mtlpp::Device::CreateSystemDefaultDevice();


    if (((int)AllDev)==0)
      printf("unable to create device\n");

      assert(device);

    printf("Selected Device %s\n",device.GetName().GetCStr());


    // device float *vOut2 [[ buffer(2) ]],
    // device float *vOut3 [[ buffer(3) ]],
    // device float *vOut4 [[ buffer(4) ]],
    // device float *vOut5 [[ buffer(5) ]],
    // device float *vOut6 [[ buffer(6) ]],
    // device float *vOut7 [[ buffer(7) ]],
    // device float *vOut8 [[ buffer(8) ]],
    // device float *vOut9 [[ buffer(9) ]],
    // device float *vOut10 [[ buffer(10) ]],
    // device float *vOut11 [[ buffer(11) ]],
    // device float *vOut12 [[ buffer(12) ]],
    // device float *vOut13 [[ buffer(13) ]],
    // device float *vOut14 [[ buffer(14) ]],
    // device float *vOut15 [[ buffer(15) ]],
    // device float *vOut16 [[ buffer(16) ]],
    // device float *vOut17 [[ buffer(17) ]],
    // device float *vOut18 [[ buffer(18) ]],
    // device float *vOut19 [[ buffer(19) ]],
    // device float *vOut20 [[ buffer(20) ]],
    // device float *vOut21 [[ buffer(21) ]],
    // device float *vOut22 [[ buffer(22) ]],
    // device float *vOut23 [[ buffer(23) ]],
    // device float *vOut24 [[ buffer(24) ]],
    // device float *vOut25 [[ buffer(25) ]],
    // device float *vOut26 [[ buffer(26) ]],
    // device float *vOut27 [[ buffer(27) ]],
    // device float *vOut28 [[ buffer(28) ]],
    // device float *vOut29 [[ buffer(29) ]],
    // device float *vOut30 [[ buffer(30) ]],
    // device float *vOut31 [[ buffer(31) ]],
    // device float *vOut32 [[ buffer(32) ]],
    // device float *vOut33 [[ buffer(33) ]],
    // device float *vOut34 [[ buffer(34) ]],
    // device float *vOut35 [[ buffer(35) ]],
    // device float *vOut36 [[ buffer(36) ]],
    // device float *vOut37 [[ buffer(37) ]],
    // device float *vOut38 [[ buffer(38) ]],
    // device float *vOut39 [[ buffer(39) ]],
    // device float *vOut40 [[ buffer(40) ]],
    // device float *vOut41 [[ buffer(41) ]],
    // device float *vOut42 [[ buffer(42) ]],
    // device float *vOut43 [[ buffer(43) ]],
    // device float *vOut44 [[ buffer(44) ]],
    // device float *vOut45 [[ buffer(45) ]],
    // device float *vOut46 [[ buffer(46) ]],
    // device float *vOut47 [[ buffer(47) ]],
    // device float *vOut48 [[ buffer(48) ]],
    // device float *vOut49 [[ buffer(49) ]],
    //
    const char shadersSrc[] = R"""(
        #include <metal_stdlib>
        using namespace metal;

        kernel void sqr(
            const device float *vIn [[ buffer(0) ]],
            device float *vOut [[ buffer(1) ]],
            device float *vOut50 [[ buffer(2) ]],
            uint id[[ thread_position_in_grid ]])
        {
            vOut[id] = vIn[id] * vIn[id] + 1000;
            vOut50[id] = vOut[id] - 333;
        }

        kernel void sqr2(
            const device float *vIn [[ buffer(0) ]],
            device float *vOut [[ buffer(1) ]],
            device float *vOut50 [[ buffer(2) ]],
            uint id[[ thread_position_in_grid ]])
        {
            vOut[id] = vIn[id] * vIn[id]*2 + 1000;
            vOut50[id] = vOut[id] - 333;
        }
    )""";

    mtlpp::Library library = device.NewLibrary(shadersSrc, mtlpp::CompileOptions(), nullptr);
    assert(library);
    mtlpp::Function sqrFunc = library.NewFunction("sqr");
    assert(sqrFunc);

    mtlpp::Function sqrFunc2 = library.NewFunction("sqr2");
    assert(sqrFunc2);

    mtlpp::ComputePipelineState computePipelineState = device.NewComputePipelineState(sqrFunc, nullptr);
    assert(computePipelineState);

    mtlpp::ComputePipelineState computePipelineState2 = device.NewComputePipelineState(sqrFunc2, nullptr);
    assert(computePipelineState2);

    mtlpp::CommandQueue commandQueue = device.NewCommandQueue();
    assert(commandQueue);

    const uint32_t dataCount = 6;

    mtlpp::Buffer inBuffer = device.NewBuffer(sizeof(float) * dataCount, mtlpp::ResourceOptions::StorageModeManaged);
    assert(inBuffer);
    mtlpp::Buffer outBuffer = device.NewBuffer(sizeof(float) * dataCount, mtlpp::ResourceOptions::StorageModeManaged);
    assert(outBuffer);

    mtlpp::Buffer outBuffer2 = device.NewBuffer(sizeof(float) * dataCount, mtlpp::ResourceOptions::StorageModeManaged);
    assert(outBuffer2);

    mtlpp::Buffer outBuffer3 = device.NewBuffer(sizeof(float) * dataCount, mtlpp::ResourceOptions::StorageModeManaged);
    assert(outBuffer3);

    mtlpp::Buffer outBuffer4 = device.NewBuffer(sizeof(float) * dataCount, mtlpp::ResourceOptions::StorageModeManaged);
    assert(outBuffer4);

    mtlpp::Buffer outBuffer5 = device.NewBuffer(sizeof(float) * dataCount, mtlpp::ResourceOptions::StorageModeManaged);
    assert(outBuffer5);


    for (uint32_t i=0; i<1000; i++)
    {
        // update input data
        {
            float* inData = static_cast<float*>(inBuffer.GetContents());
            for (uint32_t j=0; j<dataCount; j++)
                inData[j] = 10 * i + j;
            inBuffer.DidModify(ns::Range(0, sizeof(float) * dataCount));
        }

        mtlpp::CommandBuffer commandBuffer = commandQueue.CommandBuffer();
        assert(commandBuffer);

        mtlpp::CommandBuffer commandBuffer2 = commandQueue.CommandBuffer();
        assert(commandBuffer);

        mtlpp::ComputeCommandEncoder commandEncoder = commandBuffer.ComputeCommandEncoder();
        commandEncoder.SetBuffer(inBuffer, 0, 0);
        commandEncoder.SetBuffer(outBuffer, 0, 1);
        commandEncoder.SetBuffer(outBuffer2, 0, 2);
        commandEncoder.SetComputePipelineState(computePipelineState);
        commandEncoder.DispatchThreadgroups(
            mtlpp::Size(1, 1, 1),
            mtlpp::Size(dataCount, 1, 1));
        commandEncoder.EndEncoding();

        mtlpp::ComputeCommandEncoder commandEncoder2 = commandBuffer2.ComputeCommandEncoder();
        commandEncoder2.SetBuffer(inBuffer, 0, 0);
        commandEncoder2.SetBuffer(outBuffer4, 0, 1);
        commandEncoder2.SetBuffer(outBuffer5, 0, 2);
        commandEncoder2.SetComputePipelineState(computePipelineState2);
        commandEncoder2.DispatchThreadgroups(
            mtlpp::Size(1, 1, 1),
            mtlpp::Size(dataCount, 1, 1));
        commandEncoder2.EndEncoding();

        mtlpp::BlitCommandEncoder blitCommandEncoder = commandBuffer.BlitCommandEncoder();
        blitCommandEncoder.Synchronize(outBuffer);
        blitCommandEncoder.Synchronize(outBuffer2);
        blitCommandEncoder.EndEncoding();

        commandBuffer.Commit();
        commandBuffer.WaitUntilCompleted();

        mtlpp::BlitCommandEncoder blitCommandEncoder2 = commandBuffer2.BlitCommandEncoder();
        blitCommandEncoder2.Synchronize(outBuffer4);
        blitCommandEncoder2.Synchronize(outBuffer5);
        blitCommandEncoder2.EndEncoding();

        commandBuffer2.Commit();
        commandBuffer2.WaitUntilCompleted();

        // read the data
        {
            float* inData = static_cast<float*>(inBuffer.GetContents());
            float* outData = static_cast<float*>(outBuffer.GetContents());
            float* outData2 = static_cast<float*>(outBuffer2.GetContents());
            float* outData4 = static_cast<float*>(outBuffer4.GetContents());
            float* outData5 = static_cast<float*>(outBuffer5.GetContents());
            for (uint32_t j=0; j<dataCount; j++)
            {
                printf("sqr(%g) = %g, %g,  %g, %g  bla\n", inData[j], outData[j],outData2[j], outData4[j],outData5[j]);
              }
        }
    }

    return 0;
}
