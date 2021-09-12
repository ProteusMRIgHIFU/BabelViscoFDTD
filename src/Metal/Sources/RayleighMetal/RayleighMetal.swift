import Metal
import MetalPerformanceShaders
import Accelerate
import Foundation


let metallib : String = (ProcessInfo.processInfo.environment["__RayleighMetal"] ?? "the lat in the dictionary was nil!") + "/Rayleigh.metallib"


//SWIFT wrapper for METAL function for Rayleigh Integral

@available(macOS 10.13, *)
@_cdecl("ForwardSimpleMetal")
public func ForwardSimpleMetal(mr2p:        UnsafePointer<Int>,
                               c_wvnb_real: UnsafePointer<Float>, 
                               c_wvnb_imag: UnsafePointer<Float>, 
                               mr1p:        UnsafePointer<Int>,
                               r2pr:        UnsafePointer<Float>, 
                               r1pr:        UnsafePointer<Float>, 
                               a1pr:        UnsafePointer<Float>,
                               u1_real:     UnsafePointer<Float>,
                               u1_imag:     UnsafePointer<Float>,
                               deviceNamepr: UnsafePointer<CChar>,
                               py_data_u2_real: UnsafeMutablePointer<Float>, 
                               py_data_u2_imag: UnsafeMutablePointer<Float>) -> Int {
    do {
        print("Beginning")
        //let deviceName = String (cString:deviceNamepr)
        let deviceName : String = ProcessInfo.processInfo.environment["__RayleighMetalDevice"]!

        print("deviceName =" + deviceName)

        var bFound = false
        var device : MTLDevice!
        for dev in MTLCopyAllDevices() {
            if dev.name.contains(deviceName)
            {
                print("Device " + deviceName + "Found!")
                bFound = true
                device = dev
            }
            
        }

        if bFound == false {
            print("Device " + deviceName + "Not Found!")
            return 1
        }

        let commandQueue = device.makeCommandQueue()!,
            defaultLibrary = try! device.makeLibrary(filepath: metallib)

    
        let mr2Buffer = UnsafeRawPointer(mr2p)
        let c_wvnb_realBuffer = UnsafeRawPointer(c_wvnb_real)
        let c_wvnb_imagBuffer = UnsafeRawPointer(c_wvnb_imag)
        let mr1Buffer = UnsafeRawPointer(mr1p)
        let r2prBuffer = UnsafeRawPointer(r2pr)
        let r1prBuffer = UnsafeRawPointer(r1pr)
        let a1prBuffer = UnsafeRawPointer(a1pr)
        let u1_realBuffer = UnsafeRawPointer(u1_real)
        let u1_imagBuffer = UnsafeRawPointer(u1_imag)
        
        let mr2 = mr2Buffer.load(as: Int.self)   
        let mr1 = mr1Buffer.load(as: Int.self)   
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

        let ForwardSimpleMetalFunction = defaultLibrary.makeFunction(name: "ForwardSimpleMetal")!
        let computePipelineState = try device.makeComputePipelineState(function: ForwardSimpleMetalFunction)
        computeCommandEncoder.setComputePipelineState(computePipelineState)

        let mr1VectorBuffer = device.makeBuffer(bytes: mr1Buffer, length: MemoryLayout<Int>.size, options: [])
        let c_wvnb_realVectorBuffer = device.makeBuffer(bytes: c_wvnb_realBuffer, length: MemoryLayout<Float>.size, options: [])
        let c_wvnb_imagVecorBuffer = device.makeBuffer(bytes: c_wvnb_imagBuffer, length: MemoryLayout<Float>.size, options: [])
        let r2prVectorBuffer = device.makeBuffer(bytes: r2prBuffer, length: MemoryLayout<Float>.size*mr2*3, options: [])
        let r1prVectorBuffer = device.makeBuffer(bytes: r1prBuffer, length: MemoryLayout<Float>.size*mr1*3, options: [])
        let a1prVectorBuffer = device.makeBuffer(bytes: a1prBuffer, length: MemoryLayout<Float>.size*mr1, options: [])
        let u1_realVectorBuffer = device.makeBuffer(bytes: u1_realBuffer, length: MemoryLayout<Float>.size*mr1, options: [])
        let u1_imagVectorBuffer = device.makeBuffer(bytes: u1_imagBuffer, length: MemoryLayout<Float>.size*mr1, options: [])
              
        
        computeCommandEncoder.setBuffer(c_wvnb_realVectorBuffer, offset: 0, index:0)
        computeCommandEncoder.setBuffer(c_wvnb_imagVecorBuffer, offset: 0, index: 1)
        computeCommandEncoder.setBuffer(mr1VectorBuffer, offset: 0, index: 2)
        computeCommandEncoder.setBuffer(r2prVectorBuffer, offset: 0, index: 3)
        computeCommandEncoder.setBuffer(r1prVectorBuffer, offset: 0, index: 4)
        computeCommandEncoder.setBuffer(a1prVectorBuffer, offset: 0, index: 5)
        computeCommandEncoder.setBuffer(u1_realVectorBuffer, offset: 0, index: 6)
        computeCommandEncoder.setBuffer(u1_imagVectorBuffer, offset: 0, index: 7)
        
        
        let py_data_u2_realRef = UnsafeMutablePointer<Float>.allocate(capacity: mr2)
        let py_data_u2_realVectorBuffer = device.makeBuffer(bytes: py_data_u2_realRef, length: mr2*MemoryLayout<Float>.size, options: [])
        let py_data_u2_imagRef = UnsafeMutablePointer<Float>.allocate(capacity: mr2)
        let py_data_u2_imagVectorBuffer = device.makeBuffer(bytes: py_data_u2_imagRef, length:  mr2*MemoryLayout<Float>.size, options: [])
        
        computeCommandEncoder.setBuffer(py_data_u2_realVectorBuffer, offset: 0, index: 8)
        computeCommandEncoder.setBuffer(py_data_u2_imagVectorBuffer, offset: 0, index: 9)
        
        let maxTotalThreadsPerThreadgroup = computePipelineState.maxTotalThreadsPerThreadgroup
        let threadExecutionWidth = computePipelineState.threadExecutionWidth
        let width  = maxTotalThreadsPerThreadgroup / threadExecutionWidth * threadExecutionWidth
        let height = 1
        let depth  = 1
        
        // 1D
        let threadsPerGroup = MTLSize(width:width, height: height, depth: depth)
        let numThreadgroups = MTLSize(width: (mr2 + width - 1) / width, height: 1, depth: 1)
        
        computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        computeCommandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // unsafe bitcast and assigin result pointer to output
        py_data_u2_real.initialize(from: py_data_u2_realVectorBuffer!.contents().assumingMemoryBound(to: Float.self), count: mr2)
        py_data_u2_imag.initialize(from: py_data_u2_imagVectorBuffer!.contents().assumingMemoryBound(to: Float.self), count: mr2)
        
        
        free(py_data_u2_realRef)
        free(py_data_u2_imagRef)

        return 0
    } catch {
        print("\(error)")
        return 1
    }
}

