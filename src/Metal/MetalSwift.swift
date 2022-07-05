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
var mex_array:[UInt64] = []

@_cdecl("InitializeMetalDevices")
public func InitializeMetalDevices(specDevice:UnsafeRawPointer, leng:Int) -> Int {   
    particle_funcs = []
    stress_funcs = []
    mex_array = []

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
    let dummyString = NSString(bytes:specDevice, length: leng/2, encoding:String.Encoding.utf8.rawValue)
    let request = dummyString as! String
    if request != "" { // Will be fixed if I understand what is being input
        var bFound = false
        for dev in MTLCopyAllDevices() {
            print(dev.name)
            if dev.name.contains(request)
                {
                print("Specified Device Found! Selecting device...")
                bFound = true
                device = dev
                break
                }
            }
            if bFound == false
            {
                print("Specified device NOT Found!")
                return -1
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
        y = x
        y +=  "_StressKernel"
        let stress_dummy = defaultLibrary.makeFunction(name: y)!
        let stress_pipeline = try! device.makeComputePipelineState(function:stress_dummy) //Adjust try
        stress_funcs.append(stress_pipeline)
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
    var ll = MemoryLayout<UInt32>.stride * lenconstuint
    constant_buffer_uint = device.makeBuffer(length: ll, options:MTLResourceOptions.storageModeManaged)
    ll = MemoryLayout<Float32>.stride * lenconstmex
    constant_buffer_mex = device.makeBuffer(length: ll, options:MTLResourceOptions.storageModeManaged)
    return 0
}

@_cdecl("SymbolInitiation_uint")
public func SymbolInitiation_uint(index: UInt32, data: UInt32) -> Int {
    /*
    print("Initializing data", Int(data), "at index", Int(index))
    */
    constant_buffer_uint!.contents().advanced(by:(Int(index) * MemoryLayout<UInt32>.stride)).storeBytes(of:data, as:UInt32.self)
    let r:Range = (Int(index) * MemoryLayout<UInt32>.stride)..<((Int(index) + 1)*MemoryLayout<UInt32>.stride)
    constant_buffer_uint!.didModifyRange(r)
    return 0
}

@_cdecl("SymbolInitiation_mex")
public func SymbolInitiation_mex(index: UInt32, data:Float32) -> Int{
    /*
    print("Initializing data", Float(data), "at index", Int(index))
    */
    constant_buffer_mex!.contents().advanced(by:(Int(index) * MemoryLayout<Float32>.stride)).storeBytes(of:data, as:Float32.self)
    let r:Range = (Int(index) * MemoryLayout<Float32>.stride)..<((Int(index) + 1)*MemoryLayout<Float32>.stride)
    constant_buffer_mex!.didModifyRange(r)
    return 0
}

@_cdecl("BufferIndexCreator")
public func BufferIndexCreator(c_mex_type:UnsafeMutablePointer<UInt64>, c_uint_type:UInt64, length_index_mex:UInt64, length_index_uint:UInt64) -> Int {
    mex_buffer = []
    var ll = MemoryLayout<UInt64>.stride * 12
    let dummyBuffer:MTLBuffer? = device.makeBuffer(bytes:c_mex_type, length: ll, options:MTLResourceOptions.storageModeManaged)
    let swift_arr = UnsafeBufferPointer(start: dummyBuffer!.contents().assumingMemoryBound(to: UInt64.self), count: 12)
    for i in (0...11) {
        ll = MemoryLayout<Float32>.stride * Int(swift_arr[i])
        let temp:MTLBuffer? = device.makeBuffer(length: ll, options:MTLResourceOptions.storageModeManaged)
        mex_buffer.append(temp)
        mex_array.append(swift_arr[i])
    }
    dummyBuffer!.setPurgeableState(MTLPurgeableState.empty)
    ll = MemoryLayout<UInt32>.stride * Int(c_uint_type)
    uint_buffer = device.makeBuffer(length:ll, options:MTLResourceOptions.storageModeManaged)

    ll = MemoryLayout<UInt32>.stride * Int(length_index_mex) * 2
    index_mex = device.makeBuffer(length:ll, options:MTLResourceOptions.storageModeManaged)
    
    ll = MemoryLayout<UInt32>.stride * Int(length_index_uint) * 2
    index_uint = device.makeBuffer(length:ll, options:MTLResourceOptions.storageModeManaged)
    
    return 0
}

@_cdecl("IndexManipMEX")
public func IndexManipMEX(data:UInt32, data2:UInt32, index:UInt32) -> Int{
    /*
    print("Storing data", data, "at index", Int(index), "and data", data2, "at index", Int(index + 1))
    */
    var ll = Int(index) * 2 * MemoryLayout<UInt32>.stride
    index_mex!.contents().advanced(by:ll).storeBytes(of:data, as:UInt32.self)
    ll = (Int(index) * 2 + 1) * MemoryLayout<UInt32>.stride
    index_mex!.contents().advanced(by:ll).storeBytes(of:data2, as:UInt32.self)
    return 0
}
@_cdecl("IndexManipUInt")
public func IndexManipUInt(data:UInt32, data2:UInt32, index:UInt32) -> Int{
    /*
    print("Storing data", data, "at index", Int(index), "and data", data2, "at index", Int(index + 1))
    */
    var ll = Int(index) * 2 * MemoryLayout<UInt32>.stride
    index_uint!.contents().advanced(by:ll).storeBytes(of:data, as:UInt32.self)
    ll = (Int(index) * 2 + 1) * MemoryLayout<UInt32>.stride
    index_uint!.contents().advanced(by:ll).storeBytes(of:data2, as:UInt32.self)
    return 0
}
@_cdecl("IndexDidModify")
public func IndexDidModify(lenind_mex:UInt64, lenind_uint:UInt64, lenconstmex:UInt64, lenconstuint:UInt64){
    var r:Range = 0 ..< (Int(lenind_mex) * 2 + 1) * MemoryLayout<UInt32>.stride
    index_mex!.didModifyRange(r)
    r = 0 ..< (Int(lenind_uint) * 2 + 1) * MemoryLayout<UInt32>.stride
    index_uint!.didModifyRange(r)
    r = 0 ..< (Int(lenconstuint) + 1) * MemoryLayout<UInt32>.stride
    constant_buffer_uint!.didModifyRange(r)
    r = 0 ..< (Int(lenconstmex) + 1) * MemoryLayout<Float32>.stride
    constant_buffer_mex!.didModifyRange(r)
}

var floatCounter:Int = 0
@_cdecl("CompleteCopyMEX")
public func CompleteCopyMEX(size:Int, ptr:UnsafeMutablePointer<Float32>, ind:UInt64, buff:UInt64) -> Int
{
    /*
    print("Copying data into buffer:", buff, "index:", ind, "size:", size)
    let data = UnsafeBufferPointer(start:ptr, count: size)
    print(Array(data))
    */
    let ll = size * MemoryLayout<Float32>.stride
    mex_buffer[Int(buff)]!.contents().advanced(by:(Int(ind) * MemoryLayout<Float32>.stride)).copyMemory(from: ptr, byteCount:ll)
    let r : Range = (Int(ind) * MemoryLayout<Float32>.stride)..<((Int(ind) * MemoryLayout<Float32>.stride) + ll + 1)
    mex_buffer[Int(buff)]!.didModifyRange(r)
    return 0
}

@_cdecl("CompleteCopyUInt")
public func CompleteCopyUInt(size:Int, ptr:UnsafeMutablePointer<UInt32>, ind:UInt64) -> Int
{
    /*
    print("UInt Copying data into index:", ind, "size:", size)
    let data = UnsafeBufferPointer(start:ptr, count: size)
    print(Array(data))
    */
    let ll = size * MemoryLayout<UInt32>.stride
    uint_buffer!.contents().advanced(by:(Int(ind) * MemoryLayout<UInt32>.stride)).copyMemory(from: ptr, byteCount:ll)
    /*
    let testdata = UnsafeBufferPointer(start: uint_buffer!.contents().advanced(by:(Int(ind) * MemoryLayout<UInt32>.stride)).assumingMemoryBound(to: UInt32.self), count:size)
    print("UInt Data")
    print(Array(testdata))
    */
    let r : Range = (Int(ind) * MemoryLayout<UInt32>.stride)..<(Int(ind) * MemoryLayout<UInt32>.stride+ll)
    uint_buffer!.didModifyRange(r)
    return 0
}

@_cdecl("GetFloatEntries")
public func GetFloatEntries(c_mex_type: UnsafeMutablePointer<UInt64>, c_uint_type: UInt64) -> UInt64
{
    floatCounter = 0
    var ll = MemoryLayout<UInt64>.stride * 12
    let dummyBuffer:MTLBuffer? = device.makeBuffer(bytes:c_mex_type, length: ll, options:MTLResourceOptions.storageModeManaged)
    let swift_arr = UnsafeBufferPointer(start: dummyBuffer!.contents().assumingMemoryBound(to: UInt64.self), count: 12)
    for i in (0...11) {
        var r:Range = 0 ..< (Int(swift_arr[i]) + 1) * MemoryLayout<Float32>.stride
        mex_buffer[i]!.didModifyRange(r)
        floatCounter += Int(swift_arr[i]) 
    }
    dummyBuffer!.setPurgeableState(MTLPurgeableState.empty)
    var r:Range = 0 ..< (Int(c_uint_type) + 1) * MemoryLayout<UInt32>.stride
    /*
    for i in 0...11{

        let temp = UnsafeBufferPointer(start: mex_buffer[i]!.contents().assumingMemoryBound(to: Float32.self), count: Int(mex_array[i]))
        let dummy = Array(temp).map { String($0) }
        let text = dummy.joined(separator:", ")
        let file = "MEXARRAY" + String(i) + ".txt"
        if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let fileURL = dir.appendingPathComponent(file)
        do {
            try text.write(to: fileURL, atomically: false, encoding:String.Encoding.utf8)
        }
        catch {/* error handling here */}
    }
    }
    */
    return UInt64(floatCounter)
}

@_cdecl("GetMaxTotalThreadsPerThreadgroup")
public func GetMaxTotalThreadsPerThreadgroup(fun:UnsafeRawPointer, id:Int) -> UInt32{
    let dummyString = NSString(bytes:fun, length: 5, encoding:String.Encoding.utf8.rawValue)
    var index:Int!
    for name in func_names{
        if name.contains(dummyString as! String){
        index = func_names.firstIndex(of: name)!
        break
        }
    }
    if id == 0{
        return UInt32(stress_funcs[index]!.maxTotalThreadsPerThreadgroup)
    }
    else{
        return UInt32(particle_funcs[index]!.maxTotalThreadsPerThreadgroup)
    }
}
@_cdecl("GetThreadExecutionWidth") 
public func GetThreadExecutionWidth(fun:UnsafeMutablePointer<CChar>, id:Int)-> UInt32 {
    let dummyString = NSString(bytes:fun, length: 5, encoding:String.Encoding.utf8.rawValue)    
    var index:Int!
    for name in func_names{
        if name.contains(dummyString as! String){
            index = func_names.firstIndex(of: name)!
            break
        }
    }
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
    }

@_cdecl("EncodeParticle")
public func EncodeParticle(fun:UnsafeRawPointer, i:UInt32, j:UInt32, k:UInt32, x:UInt32, y:UInt32, z:UInt32){
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
}

@_cdecl("EncodeCommit")
public func EncodeCommit(){
    stress_commandBuffer.commit()
    stress_commandBuffer.waitUntilCompleted()

}

var SnapShotsBuffer:MTLBuffer?

@_cdecl("CreateAndCopyFromMXVarOnGPUSnapShot")
public func CreateAndCopyFromMXVarOnGPUSnapShot(numElements:Int, data:UnsafeMutablePointer<Float32>)
{
    let ll =  numElements * MemoryLayout<Float32>.stride
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
public func CopyFromGPUMEX(index:UInt64) -> UnsafeMutablePointer<Float32>{
    /*
    let temp = UnsafeBufferPointer(start: mex_buffer[Int(index)]!.contents().assumingMemoryBound(to: Float32.self), count: Int(mex_array[Int(index)]))
    print(Array(temp))
    */
    return mex_buffer[Int(index)]!.contents().assumingMemoryBound(to: Float32.self)
}

@_cdecl("CopyFromGPUUInt")
public func CopyFromGPUUInt() -> UnsafeMutablePointer<UInt32>{
    return uint_buffer!.contents().assumingMemoryBound(to: UInt32.self)
}

@_cdecl("CopyFromGpuSnapshot")
public func CopyFromGpuSnapshot() -> UnsafeMutablePointer<Float32>{
    return SnapShotsBuffer!.contents().assumingMemoryBound(to: Float32.self)
}

@_cdecl("maxThreadSensor")
public func maxThreadSensor() -> Int{
    return computePipelineState_Sensors.maxTotalThreadsPerThreadgroup
}

@_cdecl("freeGPUextern")
public func freeGPUextern() {
    /*
    for i in 0...11{

        let temp = UnsafeBufferPointer(start: mex_buffer[i]!.contents().assumingMemoryBound(to: Float32.self), count: Int(mex_array[i]))
        let dummy = Array(temp).map { String($0) }
        let text = dummy.joined(separator:", ")
        let file = "MEXARRAYAFTER" + String(i) + ".txt"
        if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let fileURL = dir.appendingPathComponent(file)
        do {
            try text.write(to: fileURL, atomically: false, encoding:String.Encoding.utf8)
        }
        catch {/* error handling here */}
    }
    }
    */
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