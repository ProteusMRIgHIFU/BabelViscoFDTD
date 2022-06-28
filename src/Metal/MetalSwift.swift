import Metal
import MetalPerformanceShaders
import Accelerate
import Foundation

let metallib : String = (ProcessInfo.processInfo.environment["__BabelMetal"] ?? "the lat in the dictionary was nil!") + "/Rayleigh.metallib"

// Defining global variables
var device:MTLDevice!
var commandQueue:MTLCommandQueue!
var computePipelineState_SnapShot:MTLComputePipelineState!
var computePipelineState_Sensors:MTLComputePipelineState!
var constant_buffer_mex:MTLBuffer?
var constant_buffer_uint:MTLBuffer?
var defaultLibrary:MTLLibrary!
let func_names:[String] = ["PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6", "MAIN_1"]
var particle_funcs:[MTLComputePipelineState?]! = []
var stress_funcs:[MTLComputePipelineState?]! = []


@_cdecl("InitializeMetalDevices")
public func InitializeMetalDevices(_ specDevice:String = "") -> Int {   
    
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
    defaultLibrary = try! device.makeLibrary(filepath: metallib)

  
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
    print(lenconstuint, lenconstmex)
    var ll = MemoryLayout<UInt32>.stride * lenconstuint
    constant_buffer_uint = device.makeBuffer(length: ll, options:MTLResourceOptions.storageModeManaged)
    ll = MemoryLayout<Float>.stride * lenconstmex
    constant_buffer_mex = device.makeBuffer(length: ll, options:MTLResourceOptions.storageModeManaged)
    print("Constant Buffers Created!")
    return 0
}

@_cdecl("SymbolInitiation_uint")
public func SymbolInitiation_uint(index: UInt32, data: UInt32) -> Int {
    constant_buffer_uint!.contents().advanced(by:(Int(index) * MemoryLayout<UInt32>.stride) ).storeBytes(of:data, as:UInt32.self)
    let r:Range = (Int(index) * MemoryLayout<UInt32>.stride)..<((Int(index) + 1)*MemoryLayout<UInt32>.stride)
    constant_buffer_uint!.didModifyRange(r)
    return 0
}

@_cdecl("SymbolInitiation_mex")
public func SymbolInitiation_mex(index: UInt32, data:Float32) -> Int{
    constant_buffer_mex!.contents().advanced(by:(Int(index) * MemoryLayout<Float>.stride)).storeBytes(of:data, as:Float.self)
    let r:Range = (Int(index) * MemoryLayout<Float>.stride)..<((Int(index) + 1)*MemoryLayout<Float>.stride)
    constant_buffer_mex!.didModifyRange(r)
    return 0
}

@_cdecl("BufferIndexCreator")
public func BufferIndexCreator(c_mex_type:UnsafeMutablePointer<UInt64>, c_uint_type:UInt64, length_index_mex:UInt64, length_index_uint:UInt64) -> Int {
    var ll = MemoryLayout<UInt64>.stride * 12
    let dummyBuffer:MTLBuffer? = device.makeBuffer(bytes:c_mex_type, length: ll, options:MTLResourceOptions.storageModeManaged)
    let swift_arr = UnsafeBufferPointer(start: dummyBuffer!.contents().assumingMemoryBound(to: UInt64.self), count: 12)
    print(Array(swift_arr))
    for i in (0...11) {
        ll = MemoryLayout<Float>.stride * Int(swift_arr[i])
        let temp:MTLBuffer? = device.makeBuffer(length: ll, options:MTLResourceOptions.storageModeManaged)
        mex_buffer.append(temp)
    }
    dummyBuffer!.setPurgeableState(MTLPurgeableState.empty)
    ll = MemoryLayout<UInt32>.stride * Int(c_uint_type)
    uint_buffer = device.makeBuffer(length:ll, options:MTLResourceOptions.storageModeManaged)
    ll = MemoryLayout<UInt32>.stride * Int(length_index_mex) * 2
    index_mex = device.makeBuffer(length:ll, options:MTLResourceOptions.storageModeManaged)
    ll = MemoryLayout<UInt32>.stride * Int(length_index_uint) * 2
    index_uint = device.makeBuffer(length:ll, options:MTLResourceOptions.storageModeManaged)
    print("Buffer and indices created!")
    return 0
}

@_cdecl("IndexManipMEX")
public func IndexManipMEX(data:UInt32, data2:UInt32, index:UInt32) -> Int{
    var ll = Int(index) * 2 * MemoryLayout<UInt32>.stride
    index_mex!.contents().advanced(by:ll).storeBytes(of:data, as:UInt32.self)
    ll = (Int(index) * 2 + 1) * MemoryLayout<UInt32>.stride
    index_mex!.contents().advanced(by:ll).storeBytes(of:data, as:UInt32.self)
    let r:Range = Int(index) * MemoryLayout<UInt32>.stride..<Int(index) * MemoryLayout<UInt32>.stride * 2
    index_mex!.didModifyRange(r)

return 0
}
@_cdecl("IndexManipUInt")
public func IndexManipUInt(data:UInt32, data2:UInt32, index:UInt32) -> Int{
    var ll = Int(index) * 2 * MemoryLayout<UInt32>.stride
    index_uint!.contents().advanced(by:ll).storeBytes(of:data, as:UInt32.self)
    ll = (Int(index) * 2 + 1) * MemoryLayout<UInt32>.stride
    index_uint!.contents().advanced(by:ll).storeBytes(of:data, as:UInt32.self)
    let r:Range = Int(index) * MemoryLayout<UInt32>.stride..<Int(index) * MemoryLayout<UInt32>.stride * 2
    index_uint!.didModifyRange(r)
    return 0
}

var floatCounter:UInt64 = 0
@_cdecl("CompleteCopyMEX")
public func CompleteCopyMEX(size:Int, ptr:UnsafeMutablePointer<Float>, ind:UInt64, buff:UInt64) -> Int
{
    print("Copying data into buffer:", buff, "index:", ind, "size:", size)
    let ll = size * MemoryLayout<Float>.stride
    mex_buffer[Int(buff)]!.contents().advanced(by:(Int(ind) * MemoryLayout<Float>.stride)).copyMemory(from: ptr, byteCount:ll)
    let r : Range = (Int(ind) * MemoryLayout<Float>.stride)..<((Int(ind) * MemoryLayout<Float>.stride)+ll)
    mex_buffer[Int(buff)]!.didModifyRange(r)
    floatCounter += 1
    return 0
}

@_cdecl("CompleteCopyUInt")
public func CompleteCopyUInt(size:Int, ptr:UnsafeMutablePointer<UInt32>, ind:UInt64) -> Int
{
    print("Copying data into index:", ind, "size:", size)
    let ll = size * MemoryLayout<UInt32>.stride
    uint_buffer!.contents().advanced(by:(Int(ind) * MemoryLayout<UInt32>.stride)).copyMemory(from: ptr, byteCount:ll)
    print("UInt32 Copy Success!")
    let r : Range = (Int(ind) * MemoryLayout<UInt32>.stride)..<(Int(ind) * MemoryLayout<UInt32>.stride+ll)
    uint_buffer!.didModifyRange(r)
    return 0
}

@_cdecl("GetFloatEntries")
public func GetFloatEntries() -> UInt64
{
    print("Float counter from Swift:", floatCounter)
    return floatCounter
}

@_cdecl("GetMaxTotalThreadsPerThreadgroup")
public func GetMaxTotalThreadsPerThreadgroup(fun:UnsafeRawPointer, id:Int) -> UInt32{
    var dummy = device.makeBuffer(bytes:fun, length:5, options:[])
    let swift_arr = UnsafeBufferPointer(start: dummy!.contents().assumingMemoryBound(to: CChar.self), count: 5)
    let dummyArray = Array(swift_arr)
    let dummyString = NSString(bytes:fun, length: 5, encoding:String.Encoding.utf8.rawValue)
    var index:Int!
    for name in func_names{
        if name.contains(dummyString as! String){
        index = func_names.firstIndex(of: name)!
        break
        }
    }
    dummy!.setPurgeableState(MTLPurgeableState.empty)

    if id == 0{
        return UInt32(stress_funcs[index]!.maxTotalThreadsPerThreadgroup)
    }
    else{
        return UInt32(particle_funcs[index]!.maxTotalThreadsPerThreadgroup)
    }
}
@_cdecl("GetThreadExecutionWidth") 
public func GetThreadExecutionWidth(fun:UnsafeMutablePointer<CChar>, id:Int)-> UInt32 {
    var dummy = device.makeBuffer(bytes:fun, length:5, options:[])
    let swift_arr = UnsafeBufferPointer(start: dummy!.contents().assumingMemoryBound(to: CChar.self), count: 5)
    let dummyArray = Array(swift_arr)
    let dummyString = NSString(bytes:fun, length: 5, encoding:String.Encoding.utf8.rawValue)    
    var index:Int!
    for name in func_names{
        if name.contains(dummyString as! String){
            index = func_names.firstIndex(of: name)!
            break
        }
    }
    dummy!.setPurgeableState(MTLPurgeableState.empty)
    if id == 0{
        return UInt32(stress_funcs[index]!.threadExecutionWidth)
    }
    else{
        return UInt32(particle_funcs[index]!.threadExecutionWidth)
    }
}

var stress_commandBuffer:MTLCommandBuffer!
@_cdecl("EncoderInit")
public func EncoderInit(){
    stress_commandBuffer = commandQueue.makeCommandBuffer()!
}

@_cdecl("EncodeStress")
public func EncodeStress(fun:UnsafeRawPointer, i:UInt32, j:UInt32, k:UInt32, x:UInt32, y:UInt32, z:UInt32){
    var dummy = device.makeBuffer(bytes:fun, length:5, options:[])
    let swift_arr = UnsafeBufferPointer(start: dummy!.contents().assumingMemoryBound(to: CChar.self), count: 5)
    let dummyArray = Array(swift_arr)
    let dummyString = NSString(bytes:fun, length: 5, encoding:String.Encoding.utf8.rawValue)
    var ind:Int!
    for name in func_names{
        if name.contains(dummyString as! String){
        ind = func_names.firstIndex(of: name)!
        break
        }
    }    
    let computeCommandEncoder = stress_commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setBuffer(constant_buffer_uint, offset: 0, index: 0)
    computeCommandEncoder.setBuffer(constant_buffer_mex, offset: 0, index: 1)
    computeCommandEncoder.setBuffer(index_mex, offset: 0, index: 2)
    computeCommandEncoder.setBuffer(index_uint, offset: 0, index: 3)
    computeCommandEncoder.setBuffer(uint_buffer, offset: 0, index: 4)
    for i in 0...11{
        computeCommandEncoder.setBuffer(mex_buffer[i], offset: 0, index: (5+i))
    }
    computeCommandEncoder.setComputePipelineState(stress_funcs[ind]!)
    computeCommandEncoder.dispatchThreadgroups(MTLSize(width: Int(i), height: Int(j), depth: Int(k)), threadsPerThreadgroup:MTLSize(width:Int(x), height: Int(y), depth: Int(z)))
    computeCommandEncoder.endEncoding()
    dummy!.setPurgeableState(MTLPurgeableState.empty)
    }

@_cdecl("EncodeParticle")
public func EncodeParticle(fun:UnsafeRawPointer, i:UInt32, j:UInt32, k:UInt32, x:UInt32, y:UInt32, z:UInt32){
    var dummy = device.makeBuffer(bytes:fun, length:5, options:[])
    let swift_arr = UnsafeBufferPointer(start: dummy!.contents().assumingMemoryBound(to: CChar.self), count: 5)
    let dummyArray = Array(swift_arr)
    let dummyString = NSString(bytes:fun, length: 5, encoding:String.Encoding.utf8.rawValue)
    var index:Int!
    for name in func_names{
        if name.contains(dummyString as! String){
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
    for i in 0...11{
        computeCommandEncoder.setBuffer(mex_buffer[i], offset: 0, index: (5+i))
    }
    computeCommandEncoder.setComputePipelineState(particle_funcs[index]!)
    computeCommandEncoder.dispatchThreadgroups(MTLSize(width: Int(i), height: Int(j), depth: Int(k)), threadsPerThreadgroup:MTLSize(width:Int(x), height: Int(y), depth: Int(z)))
    computeCommandEncoder.endEncoding()
    dummy!.setPurgeableState(MTLPurgeableState.empty)
}

@_cdecl("EncodeCommit")
public func EncodeCommit(){
    stress_commandBuffer.commit()
    stress_commandBuffer.waitUntilCompleted()
}

var SnapShotsBuffer:MTLBuffer?

@_cdecl("CreateAndCopyFromMXVarOnGPUSnapShot")
public func CreateAndCopyFromMXVarOnGPUSnapShot(numElements:Int, data:UnsafeMutablePointer<Float>)
{
    let ll =  numElements * MemoryLayout<Float>.stride
    SnapShotsBuffer = device.makeBuffer(bytes:data, length: ll, options: MTLResourceOptions.storageModeManaged)
}

@_cdecl("EncodeSnapShots")
public func EncodeSnapShots(i:UInt32, j:UInt32){
    let SnapShotsCommandBuffer = commandQueue.makeCommandBuffer()!
    let computeCommandEncoder = SnapShotsCommandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setBuffer(constant_buffer_uint, offset: 0, index: 0)
    computeCommandEncoder.setBuffer(constant_buffer_mex, offset: 0, index: 1)
    computeCommandEncoder.setBuffer(index_mex, offset: 0, index: 2)
    computeCommandEncoder.setBuffer(index_uint, offset: 0, index: 3)
    computeCommandEncoder.setBuffer(uint_buffer, offset: 0, index: 4)
    for i in 0...11{
        computeCommandEncoder.setBuffer(mex_buffer[i], offset: 0, index: (5+i))
    }
    computeCommandEncoder.setBuffer(SnapShotsBuffer, offset:0, index: 17)
    computeCommandEncoder.setComputePipelineState(computePipelineState_SnapShot)
    computeCommandEncoder.dispatchThreadgroups(MTLSize(width: Int(i), height: Int(j), depth: 1), threadsPerThreadgroup:MTLSize(width:8, height:8, depth:1))
    computeCommandEncoder.endEncoding()
}

@_cdecl("EncodeSensors")
public func EncodeSensors(i:UInt32, j:UInt32, k:UInt32, x:UInt32, y:UInt32, z:UInt32){
    let SensorsCommandBuffer = commandQueue.makeCommandBuffer()!
    let computeCommandEncoder = SensorsCommandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setBuffer(constant_buffer_uint, offset: 0, index: 0)
    computeCommandEncoder.setBuffer(constant_buffer_mex, offset: 0, index: 1)
    computeCommandEncoder.setBuffer(index_mex, offset: 0, index: 2)
    computeCommandEncoder.setBuffer(index_uint, offset: 0, index: 3)
    computeCommandEncoder.setBuffer(uint_buffer, offset: 0, index: 4)
    for i in 0...11{
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
  for i in 0...11{
    blitCommandEncoderSync.synchronize(resource: mex_buffer[i]!) 
  }
  blitCommandEncoderSync.endEncoding()
  commandBufferSync.commit()
  commandBufferSync.waitUntilCompleted()
  print("GPU and CPU Synced!")
}

@_cdecl("CopyFromGPUMEX")
public func CopyFromGPUMEX(index:UInt64) -> UnsafeMutablePointer<Float>{
  return mex_buffer[Int(index)]!.contents().assumingMemoryBound(to: Float.self)
}

@_cdecl("CopyFromGPUUInt")
public func CopyFromGPUUInt() -> UnsafeMutablePointer<UInt32>{
  return uint_buffer!.contents().assumingMemoryBound(to: UInt32.self)
}

@_cdecl("CopyFromGpuSnapshot")
public func CopyToGpuSnapshot() -> UnsafeMutablePointer<Float>{
  return SnapShotsBuffer!.contents().assumingMemoryBound(to: Float.self)
}

@_cdecl("maxThreadSensor")
public func maxThreadSensor() -> Int{
  print("Getting max Total Threads for Sensor")
  return computePipelineState_Sensors.maxTotalThreadsPerThreadgroup
}

@_cdecl("freeGPUextern")
public func freeGPUextern() {
    constant_buffer_uint!.setPurgeableState(MTLPurgeableState.empty)
    constant_buffer_mex!.setPurgeableState(MTLPurgeableState.empty)
    for i in 0...11{
        mex_buffer[i]!.setPurgeableState(MTLPurgeableState.empty)
    }
    uint_buffer!.setPurgeableState(MTLPurgeableState.empty)
    index_mex!.setPurgeableState(MTLPurgeableState.empty)
    index_uint!.setPurgeableState(MTLPurgeableState.empty)
    SnapShotsBuffer!.setPurgeableState(MTLPurgeableState.empty)
}