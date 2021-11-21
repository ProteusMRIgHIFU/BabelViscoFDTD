import Metal
import MetalPerformanceShaders
import Accelerate
import Foundation


let metallib : String = (ProcessInfo.processInfo.environment["__RayleighMetal"] ?? "the lat in the dictionary was nil!") + "/Rayleigh.metallib"


@available(macOS 10.13, *)
@_cdecl("PrintMetalDevices")
public func PrintMetalDevices() -> Int {
    let devices = MTLCopyAllDevices()
    print("Metal devices")
    for device in devices {
        print(device.name)
        print("Is device low power? \(device.isLowPower).")
        print("Is device external? \(device.isRemovable).")
        print("Maximum threads per group: \(device.maxThreadsPerThreadgroup).")
        print("Maximum buffer length: \(Float(device.maxBufferLength) / 1024 / 1024 / 1024) GB.")
    }
    return 0
}
//SWIFT wrapper for METAL function for Rayleigh Integral

@available(macOS 10.13, *)
@_cdecl("ForwardSimpleMetal")
public func ForwardSimpleMetal(mr2p:        UnsafeMutablePointer<Int>,
                               c_wvnb_real: UnsafeMutablePointer<Float>, 
                               c_wvnb_imag: UnsafeMutablePointer<Float>, 
                               mr1p:        UnsafeMutablePointer<Int>,
                               r2pr:        UnsafeMutablePointer<Float>, 
                               r1pr:        UnsafeMutablePointer<Float>, 
                               a1pr:        UnsafeMutablePointer<Float>,
                               u1_real:     UnsafeMutablePointer<Float>,
                               u1_imag:     UnsafeMutablePointer<Float>,
                               deviceNamepr: UnsafePointer<CChar>,
                               py_data_u2_real: UnsafeMutablePointer<Float>, 
                               py_data_u2_imag: UnsafeMutablePointer<Float>,
                               bUseAlignedMemp: UnsafeMutablePointer<Int>) -> Int {
    do {
        // print("Beginning")
        //let deviceName = String (cString:deviceNamepr)
        let deviceName : String = ProcessInfo.processInfo.environment["__RayleighMetalDevice"]!

        // print("deviceName =" + deviceName)

        var bFound = false
        var device : MTLDevice!
        for dev in MTLCopyAllDevices() {
            if dev.name.contains(deviceName)
            {
                // print("Device " + deviceName + "Found!")
                bFound = true
                device = dev
            }
            
        }

        if bFound == false {
            // print("Device " + deviceName + "Not Found!")
            return 1
        }

        let commandQueue = device.makeCommandQueue()!,
            defaultLibrary = try! device.makeLibrary(filepath: metallib)

        let mr1Buffer = UnsafeMutableRawPointer(mr1p)
        let mr2Buffer = UnsafeMutableRawPointer(mr2p)
        let bUseAlignedMemBuffer = UnsafeMutableRawPointer(bUseAlignedMemp)
        let bUseAlignedMem = bUseAlignedMemBuffer.load(as:Int.self)
        // if bUseAlignedMem == 1 {
        //     print("Uising aligned memory")
        // }

        let c_wvnb_realBuffer = UnsafeMutableRawPointer(c_wvnb_real)
        let c_wvnb_imagBuffer = UnsafeMutableRawPointer(c_wvnb_imag)
        
        let r2prBuffer = UnsafeMutableRawPointer(r2pr)
        let r1prBuffer =  UnsafeMutableRawPointer(r1pr)
        let a1prBuffer = UnsafeMutableRawPointer(a1pr)
        let u1_realBuffer = UnsafeMutableRawPointer(u1_real)
        let u1_imagBuffer = UnsafeMutableRawPointer(u1_imag)
        
        let mr2 = mr2Buffer.load(as: Int.self)   
        let mr1 = mr1Buffer.load(as: Int.self) 
       
        var ll = MemoryLayout<Float>.size*mr1*3
        let PAGE_MAP_SIZE = 16384
        if bUseAlignedMem == 1
        {
            if ll % PAGE_MAP_SIZE != 0 {
                ll = (Int(ll/PAGE_MAP_SIZE)+1)*PAGE_MAP_SIZE
            }
        }
        var r1prVectorBuffer:MTLBuffer?
        if bUseAlignedMem == 1
        { 
            r1prVectorBuffer = device.makeBuffer(bytesNoCopy:r1prBuffer, length: ll, options: [],deallocator:nil)
        }
        else {
            r1prVectorBuffer = device.makeBuffer(bytes: r1prBuffer, length: MemoryLayout<Float>.size*mr1*3, options: [])
        }
       
        let mr1VectorBuffer = device.makeBuffer(bytes: mr1Buffer, length: MemoryLayout<Int>.size, options: [])
        let c_wvnb_realVectorBuffer = device.makeBuffer(bytes: c_wvnb_realBuffer, length: MemoryLayout<Float>.size, options: [])
        let c_wvnb_imagVecorBuffer = device.makeBuffer(bytes: c_wvnb_imagBuffer, length: MemoryLayout<Float>.size, options: [])
        
        var r2prVectorBuffer:MTLBuffer?
        ll = MemoryLayout<Float>.size*mr2*3
        if bUseAlignedMem == 1
        {
            if ll % PAGE_MAP_SIZE != 0 {
                ll = (Int(ll/PAGE_MAP_SIZE)+1)*PAGE_MAP_SIZE
            }
        }
        if bUseAlignedMem == 1
        { 
            r2prVectorBuffer = device.makeBuffer(bytesNoCopy:r2prBuffer, length: ll, options: [],deallocator:nil)
            
        }
        else {
            r2prVectorBuffer = device.makeBuffer(bytes: r2prBuffer, length: ll, options: [])
        }

        var a1prVectorBuffer:MTLBuffer?
        var u1_realVectorBuffer:MTLBuffer?
        var u1_imagVectorBuffer:MTLBuffer?
        ll = MemoryLayout<Float>.size*mr1
        if bUseAlignedMem == 1
        {
            if ll % PAGE_MAP_SIZE != 0 {
                ll = (Int(ll/PAGE_MAP_SIZE)+1)*PAGE_MAP_SIZE
            }
        }
        if bUseAlignedMem == 1
        { 
            a1prVectorBuffer = device.makeBuffer(bytesNoCopy:a1prBuffer, length: ll, options: [],deallocator:nil)
            u1_realVectorBuffer = device.makeBuffer(bytesNoCopy:u1_realBuffer, length: ll, options: [],deallocator:nil)
            u1_imagVectorBuffer = device.makeBuffer(bytesNoCopy:u1_imagBuffer, length: ll, options: [],deallocator:nil)
            
        }
        else {
            a1prVectorBuffer = device.makeBuffer(bytes: a1prBuffer, length: ll, options: [])
            u1_realVectorBuffer = device.makeBuffer(bytes: u1_realBuffer, length: MemoryLayout<Float>.size*mr1, options: [])
            u1_imagVectorBuffer = device.makeBuffer(bytes: u1_imagBuffer, length: MemoryLayout<Float>.size*mr1, options: [])
        }

        ll = MemoryLayout<Float>.size*mr2
        if bUseAlignedMem == 1
        {
            if ll % PAGE_MAP_SIZE != 0 {
                ll = (Int(ll/PAGE_MAP_SIZE)+1)*PAGE_MAP_SIZE
            }
        }
        var py_data_u2_realRef = UnsafeMutablePointer(py_data_u2_real)
        var py_data_u2_realVectorBuffer:MTLBuffer?
        var py_data_u2_imagRef = UnsafeMutablePointer(py_data_u2_imag) 
        var py_data_u2_imagVectorBuffer:MTLBuffer?
        if bUseAlignedMem == 1 {
            py_data_u2_realVectorBuffer = device.makeBuffer(bytesNoCopy:py_data_u2_realRef, length: ll, options: [],deallocator:nil)
            py_data_u2_imagVectorBuffer = device.makeBuffer(bytesNoCopy:py_data_u2_imagRef, length: ll, options: [],deallocator:nil)
            
        }
        else {

            py_data_u2_realRef = UnsafeMutablePointer<Float>.allocate(capacity: mr2)
            py_data_u2_imagRef = UnsafeMutablePointer<Float>.allocate(capacity: mr2)
            py_data_u2_realVectorBuffer = device.makeBuffer(bytes: py_data_u2_realRef, length: mr2*MemoryLayout<Float>.size, options: [])
            py_data_u2_imagVectorBuffer = device.makeBuffer(bytes: py_data_u2_imagRef, length:  mr2*MemoryLayout<Float>.size, options: [])

        }
        
        // We need to split in small chunks to be sure the kernel does not take too much time
        // otherwise the OS will kill it
        var nm2 = Int(0)
        let TotalIteractions = mr1 * mr2 
        let NonBlockingstep = Int(500e6)
        var nm_step = Int(NonBlockingstep/mr1)


        if nm_step > mr2
        {
            nm_step=mr2
        }

        var basemr2 = Int(0)
        var n2Limit = Int(0)
        while basemr2 < mr2
        {   

            if basemr2+nm_step<mr2
            {
                n2Limit = nm_step
            }
            else
            {
                n2Limit = mr2 - basemr2
            }

            let commandBuffer = commandQueue.makeCommandBuffer()!
            let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

            let ForwardSimpleMetalFunction = defaultLibrary.makeFunction(name: "ForwardSimpleMetal")!
            let computePipelineState = try device.makeComputePipelineState(function: ForwardSimpleMetalFunction)
            computeCommandEncoder.setComputePipelineState(computePipelineState)


            let mr2VectorBuffer = device.makeBuffer(bytes: &n2Limit, length: MemoryLayout<Int>.size, options: [])
            computeCommandEncoder.setBuffer(c_wvnb_realVectorBuffer, offset: 0, index:0)
            computeCommandEncoder.setBuffer(c_wvnb_imagVecorBuffer, offset: 0, index: 1)
            computeCommandEncoder.setBuffer(mr1VectorBuffer, offset: 0, index: 2)
            computeCommandEncoder.setBuffer(mr2VectorBuffer, offset: 0, index: 3)
            computeCommandEncoder.setBuffer(r2prVectorBuffer, 
                            offset: MemoryLayout<Float>.size*basemr2*3, index: 4)
            computeCommandEncoder.setBuffer(r1prVectorBuffer, offset: 0, index: 5)
            computeCommandEncoder.setBuffer(a1prVectorBuffer, offset: 0, index: 6)
            computeCommandEncoder.setBuffer(u1_realVectorBuffer, offset: 0, index: 7)
            computeCommandEncoder.setBuffer(u1_imagVectorBuffer, offset: 0, index: 8)
            computeCommandEncoder.setBuffer(py_data_u2_realVectorBuffer, offset: MemoryLayout<Float>.size*basemr2, index: 9)
            computeCommandEncoder.setBuffer(py_data_u2_imagVectorBuffer, offset: MemoryLayout<Float>.size*basemr2, index: 10)
            
            let maxTotalThreadsPerThreadgroup = computePipelineState.maxTotalThreadsPerThreadgroup
            let threadExecutionWidth = computePipelineState.threadExecutionWidth
            let width  = maxTotalThreadsPerThreadgroup / threadExecutionWidth * threadExecutionWidth
            let height = 1
            let depth  = 1
            
            // 1D
            let threadsPerGroup = MTLSize(width:width, height: height, depth: depth)
            let numThreadgroups = MTLSize(width: (n2Limit + width - 1) / width, height: 1, depth: 1)
            
            computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
            computeCommandEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            basemr2+=nm_step
        }
        // unsafe bitcast and assigin result pointer to output
        if bUseAlignedMem == 0 
        {
            py_data_u2_real.initialize(from: py_data_u2_realVectorBuffer!.contents().assumingMemoryBound(to: Float.self), count: mr2)
            py_data_u2_imag.initialize(from: py_data_u2_imagVectorBuffer!.contents().assumingMemoryBound(to: Float.self), count: mr2)
            free(py_data_u2_realRef)
            free(py_data_u2_imagRef)
        }
        
        

        return 0
    } catch {
        print("\(error)")
        return 1
    }
}

