import Metal
import MetalPerformanceShaders
import Accelerate
import Foundation

// Defining global variables
var device:MTLDevice!
var commandQueue:MTLCommandQueue!
var computePipelineState_SnapShot:MTLComputePipelineState!
var computePipelineState_Sensors:MTLComputePipelineState!
var constant_buffer_mex:MTLBuffer?
var constant_buffer_uint:MTLBuffer?
var defaultLibrary:MTLLibrary!
let func_names:[String] = ["PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6", "MAIN_1"]
var particle_funcs:[MTLComputePipelineState]!
var stress_funcs:[MTLComputePipelineState]!

@_cdecl("InitializeMetalDevices")
public func InitializeMetalDevices(_ specDevice:String,_ library:String = "asdasdasdasdasdasd") -> Int {   
    let devices = MTLCopyAllDevices()
    print("Found", devices.count, "METAL Devices!")
    if devices.count == 0 {
        print("No devices found!")
        return -1
    }
    print("Devices:")
    for d in devices {
        print(d.name)
        print("Is device low power? \(d.isLowPower).")
        print("Is device external? \(d.isRemovable).")
        print("Maximum threads per group: \(d.maxThreadsPerThreadgroup).")
        print("Maximum buffer length: \(Float(d.maxBufferLength) / 1024 / 1024 / 1024) GB.")
        print("")
    }

    print("The Device METAL selects as default is:", MTLCreateSystemDefaultDevice()!.name)
  
    if specDevice != "asdasdasdasdasdasd" { // Will be fixed if I understand what is being input
        var bFound = false
        for dev in MTLCopyAllDevices() {
        if dev.name.contains("BOBOBOBOBOB")
            {
            print("Specified Device Found! Selecting device...")
            bFound = true
            device = dev
            break
            }
        }
        if bFound == false
        {
            print("Specified device NOT Found! Defaulting to system default device.")
            device = MTLCreateSystemDefaultDevice()!

        }
    }
    else{
        print("No device specified, defaulting to system default device.")
        device = MTLCreateSystemDefaultDevice()!
    }
    print("Making Command Queue and Library...")
    commandQueue = device.makeCommandQueue()!
    defaultLibrary = try! device.makeLibrary(filepath: "metal.metallib")

  
    print("Getting Particle and Stress Functions...") // Hopefully there's a better way to do this?
    for x in func_names
    {
        var y = x
        y +=  "_ParticleKernel"
        let particle_dummy = defaultLibrary.makeFunction(name: y)!
        let particle_pipeline = try! device.makeComputePipelineState(function:particle_dummy) //Adjust try
        particle_funcs.append(particle_pipeline)
        print(y, "function and pipeline created!")
        y = x
        y +=  "_StressKernel"
        let stress_dummy = defaultLibrary.makeFunction(name: y)!
        let stress_pipeline = try! device.makeComputePipelineState(function:stress_dummy) //Adjust try
        stress_funcs.append(stress_pipeline)
        print(y, "function and pipeline created!")
    }

    print("Making Compute Pipeline State Objects for SnapShot and Sensors...")

    let SnapShotFunc = defaultLibrary.makeFunction(name: "SnapShot")!
    let SensorsKernelFunc = defaultLibrary.makeFunction(name: "SensorsKernel")!

    computePipelineState_SnapShot = try! device.makeComputePipelineState(function:SnapShotFunc) // Change try! so that we can throw a integer error
    computePipelineState_Sensors = try! device.makeComputePipelineState(function:SensorsKernelFunc) // Change try! so that we can throw a integer error

    print("Function creation success!")
    return 0
}

var mex_buffer:[MTLBuffer?] = []
var uint_buffer:MTLBuffer?
var index_mex:MTLBuffer?
var index_uint:MTLBuffer?

@_cdecl("ConstantBuffers")
public func ConstantBuffers(lenconstuint: Int, lenconstmex: Int) -> Int
{
    var ll = MemoryLayout<UInt>.stride * lenconstuint
    constant_buffer_uint = device.makeBuffer(length: ll, options:MTLResourceOptions.storageModeManaged)
    ll = MemoryLayout<Float>.stride * lenconstmex
    constant_buffer_mex = device.makeBuffer(length: ll, options:MTLResourceOptions.storageModeManaged)
    return 0
}

@_cdecl("SymbolInitiation_uint")
public func SymbolInitiation_uint(index: UInt, data: UInt) -> Int {
    constant_buffer_uint!.contents().advanced(by:(Int(index))).storeBytes(of:data, as:UInt.self)
    let r:Range = (Int(index) * MemoryLayout<UInt>.stride)..<((Int(index) + 1)*MemoryLayout<UInt>.stride)
    constant_buffer_uint!.didModifyRange(r)
    return 0
}

@_cdecl("SymbolInitiation_mex")
public func SymbolInitiation_mex(index: UInt, data:Float) -> Int{
    constant_buffer_mex!.contents().advanced(by:(Int(index))).storeBytes(of:data, as:Float.self)
    let r:Range = (Int(index) * MemoryLayout<Float>.stride)..<((Int(index) + 1)*MemoryLayout<Float>.stride)
    constant_buffer_mex!.didModifyRange(r)
    return 0
}

@_cdecl("BufferIndexCreator")
public func bufferindexcreator(c_mex_type:UnsafeMutablePointer<ULONG>, c_uint_type:ULONG, length_index_mex:ULONG, length_index_uint:ULONG) -> Int {
    var ll = MemoryLayout<ULONG>.stride * 12
    let dummyBuffer:MTLBuffer? = device.makeBuffer(bytes:c_mex_type, length: ll, options:MTLResourceOptions.storageModeManaged)
    let swift_arr = UnsafeBufferPointer(start: dummyBuffer!.contents().assumingMemoryBound(to: Int.self), count: 12)
    for i in 0...11{
        ll = MemoryLayout<Float>.stride * swift_arr[i]
        let temp:MTLBuffer? = device.makeBuffer(length: ll, options:MTLResourceOptions.storageModeManaged)
        mex_buffer.append(temp)
    }
    ll = MemoryLayout<UInt>.stride * Int(c_uint_type)
    uint_buffer = device.makeBuffer(length:ll, options:MTLResourceOptions.storageModeManaged)
    ll = MemoryLayout<UInt>.stride * Int(length_index_mex) * 2
    index_mex = device.makeBuffer(length:ll, options:MTLResourceOptions.storageModeManaged)
    ll = MemoryLayout<UInt>.stride * Int(length_index_uint) * 2
    index_uint = device.makeBuffer(length:ll, options:MTLResourceOptions.storageModeManaged)
    return 0
}

@_cdecl("IndexManipMEX")
public func IndexManipMEX(data:UInt, data2:UInt, index:UInt) -> Int{
    var ll = Int(index) * MemoryLayout<UInt>.stride
    index_mex!.contents().advanced(by:ll).storeBytes(of:data, as:UInt.self)
    ll = (Int(index) + 1) * MemoryLayout<UInt>.stride
    index_mex!.contents().advanced(by:ll).storeBytes(of:data, as:UInt.self)
    let r:Range = Int(index) * MemoryLayout<UInt>.stride..<Int(index) * MemoryLayout<UInt>.stride * 2
    index_mex!.didModifyRange(r)

return 0
}
@_cdecl("IndexManipUInt")
public func IndexManipUInt(data:UInt, data2:UInt, index:UInt) -> Int{
    var ll = Int(index) * MemoryLayout<Float>.stride
    index_uint!.contents().advanced(by:ll).storeBytes(of:data, as:UInt.self)
    ll = (Int(index) + 1) * MemoryLayout<Float>.stride
    index_uint!.contents().advanced(by:ll).storeBytes(of:data, as:UInt.self)
    let r:Range = Int(index) * MemoryLayout<UInt>.stride..<Int(index) * MemoryLayout<UInt>.stride * 2
    index_uint!.didModifyRange(r)
    return 0
}

var floatCounter:ULONG = 0
@_cdecl("CompleteCopyMEX")
public func CompleteCopyMEX(size:Int, ptr:UnsafeMutablePointer<Float>, ind:UInt, buff:UInt) -> Int
{
    let ll = size * MemoryLayout<Float>.stride
    mex_buffer[Int(buff)]!.contents().advanced(by:(Int(ind) * MemoryLayout<Float>.stride)).copyMemory(from: ptr, byteCount:ll)
    let r : Range = (Int(ind) * MemoryLayout<UInt>.stride)..<((Int(ind) * MemoryLayout<UInt>.stride)+ll)
    mex_buffer[Int(buff)]!.didModifyRange(r)
    floatCounter += 1
    return 0
}

@_cdecl("CompleteCopyUInt")
public func CompleteCopyUInt(size:Int, ptr:UnsafeMutablePointer<UInt>, ind:UInt) -> Int
{
    let ll = size * MemoryLayout<UInt>.stride
    uint_buffer!.contents().advanced(by:(Int(ind) * MemoryLayout<UInt>.stride)).copyMemory(from: ptr, byteCount:ll)
    let r : Range = (Int(ind) * MemoryLayout<UInt>.stride)..<(Int(ind) * MemoryLayout<UInt>.stride+ll)
    uint_buffer!.didModifyRange(r)
    return 0
}

@_cdecl("GetFloatEntries")
public func GetFloatEntries() -> ULONG
{
    return floatCounter
}

@_cdecl("GetMaxTotalThreadsPerThreadgroup")
public func GetMaxTotalThreadsPerThreadgroup(fun:String, id:Int) -> UInt{
    guard id > 1 else {
        print("Something went horribly wrong.")
        return 0 // Change this
    }
    var index:Int!
    for name in func_names{
        if name.contains(fun){
        index = func_names.firstIndex(of: name)!
        break
        }
    }
    if index == 0{
        return UInt(stress_funcs[index].maxTotalThreadsPerThreadgroup)
    }
    else if index == 1{
        return UInt(stress_funcs[index].maxTotalThreadsPerThreadgroup)
    }
  return 0
}
@_cdecl("GetThreadExecutionWidth") 
public func GetThreadExecutionWidth(fun:String, id:Int)-> UInt {
    guard id > 1 else {
        print("Something went horribly wrong.")
        return 0 // Change this
    }
    var index:Int!
    for name in func_names{
        if name.contains(fun){
            index = func_names.firstIndex(of: name)!
            break
        }
    }
    if index == 0{
        return UInt(stress_funcs[index].threadExecutionWidth)
    }
    else if index == 1{
        return UInt(stress_funcs[index].threadExecutionWidth)
    }
  return 0
}

var stress_commandBuffer:MTLCommandBuffer!
@_cdecl("EncodeInit")
public func EncodeInit(){
    stress_commandBuffer = commandQueue.makeCommandBuffer()!
}

@_cdecl("EncodeStress")
public func EncodeStress(fun:String, i:UInt, j:UInt, k:UInt, x:UInt, y:UInt, z:UInt){
    var index:Int!
    for name in func_names{
        if name.contains(fun){
            index = func_names.firstIndex(of: name)!
            break
        }
    }
    let computeCommandEncoder = stress_commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setBuffer(constant_buffer_uint, offset: 0, index: 0)
    computeCommandEncoder.setBuffer(constant_buffer_mex, offset: 0, index: 1)
    computeCommandEncoder.setBuffer(index_mex, offset: 0, index: 2)
    computeCommandEncoder.setBuffer(index_uint, offset: 0, index: 3)
    computeCommandEncoder.setBuffer(uint_buffer, offset: 0, index: 4)
    for i in 0...12{
        computeCommandEncoder.setBuffer(mex_buffer[i], offset: 0, index: (5+i))
    }
    computeCommandEncoder.setComputePipelineState(stress_funcs[index])
    computeCommandEncoder.dispatchThreadgroups(MTLSize(width: Int(i), height: Int(j), depth: Int(k)), threadsPerThreadgroup:MTLSize(width:Int(x), height: Int(y), depth: Int(z)))
    computeCommandEncoder.endEncoding()
}

@_cdecl("EncodeParticle")
public func EncodeParticle(fun:String, i:UInt, j:UInt, k:UInt, x:UInt, y:UInt, z:UInt){
    var index:Int!
    for name in func_names{
        if name.contains(fun){
            index = func_names.firstIndex(of: name)!
            break
        }
    }
    let computeCommandEncoder = stress_commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setBuffer(constant_buffer_uint, offset: 0, index: 0)
    computeCommandEncoder.setBuffer(constant_buffer_mex, offset: 0, index: 1)
    computeCommandEncoder.setBuffer(index_mex, offset: 0, index: 2)
    computeCommandEncoder.setBuffer(index_uint, offset: 0, index: 3)
    computeCommandEncoder.setBuffer(uint_buffer, offset: 0, index: 4)
    for i in 0...12{
        computeCommandEncoder.setBuffer(mex_buffer[i], offset: 0, index: (5+i))
    }
    computeCommandEncoder.setComputePipelineState(particle_funcs[index])
    computeCommandEncoder.dispatchThreadgroups(MTLSize(width: Int(i), height: Int(j), depth: Int(k)), threadsPerThreadgroup:MTLSize(width:Int(x), height: Int(y), depth: Int(z)))
    computeCommandEncoder.endEncoding()
}

@_cdecl("EncodeCommit")
public func EncodeCommit(){
    stress_commandBuffer.commit()
    stress_commandBuffer.waitUntilCompleted()
}

var SnapShotsBuffer:MTLBuffer?

@_cdecl("CreateAndCopyFromMXVarOnGPU2")
public func CreateAndCopyFromMXVarOnGPU(numElements:Int, data:UnsafeMutablePointer<Float>)
{
    let ll =  numElements * MemoryLayout<Float>.stride
    SnapShotsBuffer = device.makeBuffer(bytes:data, length: ll, options: MTLResourceOptions.storageModeManaged)
}


@_cdecl("EncodeSnapShots")
public func EncodeSnapShots(i:UInt, j:UInt){
    let SnapShotsCommandBuffer = commandQueue.makeCommandBuffer()!
    let computeCommandEncoder = SnapShotsCommandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setBuffer(constant_buffer_uint, offset: 0, index: 0)
    computeCommandEncoder.setBuffer(constant_buffer_mex, offset: 0, index: 1)
    computeCommandEncoder.setBuffer(index_mex, offset: 0, index: 2)
    computeCommandEncoder.setBuffer(index_uint, offset: 0, index: 3)
    computeCommandEncoder.setBuffer(uint_buffer, offset: 0, index: 4)
    for i in 0...12{
        computeCommandEncoder.setBuffer(mex_buffer[i], offset: 0, index: (5+i))
    }
    computeCommandEncoder.setBuffer(SnapShotsBuffer, offset:0, index: 17)
    computeCommandEncoder.setComputePipelineState(computePipelineState_SnapShot)
    computeCommandEncoder.dispatchThreadgroups(MTLSize(width: Int(i), height: Int(j), depth: 1), threadsPerThreadgroup:MTLSize(width:8, height:8, depth:1))
    computeCommandEncoder.endEncoding()
}

@_cdecl("EncodeSensors")
public func EncodeSensors(i:UInt, j:UInt, k:UInt, x:UInt, y:UInt, z:UInt){
    let SensorsCommandBuffer = commandQueue.makeCommandBuffer()!
    let computeCommandEncoder = SensorsCommandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setBuffer(constant_buffer_uint, offset: 0, index: 0)
    computeCommandEncoder.setBuffer(constant_buffer_mex, offset: 0, index: 1)
    computeCommandEncoder.setBuffer(index_mex, offset: 0, index: 2)
    computeCommandEncoder.setBuffer(index_uint, offset: 0, index: 3)
    computeCommandEncoder.setBuffer(uint_buffer, offset: 0, index: 4)
    for i in 0...12{
        computeCommandEncoder.setBuffer(mex_buffer[i], offset: 0, index: (5+i))
    }
    computeCommandEncoder.setComputePipelineState(computePipelineState_Sensors)
    computeCommandEncoder.dispatchThreadgroups(MTLSize(width: Int(i), height: Int(j), depth: Int(k)), threadsPerThreadgroup:MTLSize(width:Int(x), height:Int(y), depth:Int(z)))
    computeCommandEncoder.endEncoding()
    SensorsCommandBuffer.commit()
    SensorsCommandBuffer.waitUntilCompleted()
}

@_cdecl("SyncChange")
public func SyncChange(){
  let commandBufferSync = commandQueue.makeCommandBuffer()!
  let blitCommandEncoderSync: MTLBlitCommandEncoder = commandBufferSync.makeBlitCommandEncoder()!
  for i in 0...12{
    blitCommandEncoderSync.synchronize(resource: mex_buffer[i]!) 
  }
  commandBufferSync.commit()
  commandBufferSync.waitUntilCompleted()
}

@_cdecl("CopyFromGpuMEX")
public func CopyToGpuMEX(index:ULONG) -> UnsafeMutablePointer<Float>{
  return mex_buffer[Int(index)]!.contents().assumingMemoryBound(to: Float.self)
}

@_cdecl("CopyFromGpuUInt")
public func CopyToGpuUInt() -> UnsafeMutablePointer<UInt>{
  return uint_buffer!.contents().assumingMemoryBound(to: UInt.self)
}

@_cdecl("CreateAndCopyFromMXVarOnGPUSensor")
public func CreateAndCopyFromMXVarOnGPUSensor(numElements:Int, data:UnsafeMutablePointer<Float>)
{
  let ll =  numElements * MemoryLayout<Float>.stride
  SnapShotsBuffer = device.makeBuffer(bytes:data, length: ll, options: MTLResourceOptions.storageModeManaged)
}

@_cdecl("maxThreadsSensor")
public func maxThreadsSensor() -> Int{
  return computePipelineState_Sensors.maxTotalThreadsPerThreadgroup
}