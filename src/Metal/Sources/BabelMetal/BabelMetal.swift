import Metal
import MetalPerformanceShaders
import Accelerate
import Foundation

let metallib : String = (ProcessInfo.processInfo.environment["__BabelMetal"] ?? "the lat in the dictionary was nil!") + "/Babel.metallib"


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

var captureManager : MTLCaptureManager!
var deviceCapture : MTLDevice!
@available(macOS 10.13, *)
@_cdecl("StartCapture")
public func StartCapture() -> Int {

    let deviceName : String = ProcessInfo.processInfo.environment["__BabelMetalDevice"]!

    // print("deviceName =" + deviceName)

    var bFound = false
    for dev in MTLCopyAllDevices() {
        if dev.name.contains(deviceName)
        {
            bFound = true
            deviceCapture = dev
        }
        
    }

    if bFound == false {
        return -1
    }

	captureManager = MTLCaptureManager.shared()
		
	guard captureManager.supportsDestination(.gpuTraceDocument) else {
		print("Capture to a GPU tracefile is not supported")
		return -1
	}
	
	// Write file to tmp folder
	let destURL = URL(fileURLWithPath: "./frameCapture.gputrace")
	
	// Set up the capture destiptor
	let captureDescriptor = MTLCaptureDescriptor()
	captureDescriptor.captureObject = deviceCapture
	captureDescriptor.destination = .gpuTraceDocument
	captureDescriptor.outputURL = destURL
		
	do {
		try captureManager.startCapture(with: captureDescriptor)
	}  catch let e {
		print("Failed to capture frame for debug: \(e.localizedDescription)")
		return -1
	}
    return 0
}

@available(macOS 10.13, *)
@_cdecl("Stopcapture")
public func Stopcapture() -> Int {

	captureManager.stopCapture()
    return 0
}

@available(macOS 10.13, *)
@_cdecl("ForwardSimpleMetal")
public func ForwardSimpleMetal(mr2p:        UnsafeMutablePointer<Int>,
                               c_wvnb_real: UnsafeMutablePointer<Float>, 
                               c_wvnb_imag: UnsafeMutablePointer<Float>,
                               MaxDistance: UnsafeMutablePointer<Float>, 
                               mr1p:        UnsafeMutablePointer<Int>,
                               r2pr:        UnsafeMutablePointer<Float>, 
                               r1pr:        UnsafeMutablePointer<Float>, 
                               a1pr:        UnsafeMutablePointer<Float>,
                               u1_real:     UnsafeMutablePointer<Float>,
                               u1_imag:     UnsafeMutablePointer<Float>,
                               deviceNamepr: UnsafePointer<CChar>,
                               py_data_u2_real: UnsafeMutablePointer<Float>, 
                               py_data_u2_imag: UnsafeMutablePointer<Float>,
                               bUseAlignedMemp: UnsafeMutablePointer<Int>,
                               u0stepsp: UnsafeMutablePointer<Int>) -> Int {
    do {
        // print("Rayleigh: Beginning")
        let deviceName : String = ProcessInfo.processInfo.environment["__BabelMetalDevice"]!

        // print("Rayleigh: deviceName =" + deviceName)

        var bFound = false
        var device : MTLDevice!
        for dev in MTLCopyAllDevices() {
            if dev.name.contains(deviceName)
            {
                // print("Rayleigh: Device " + deviceName + "Found!")
                bFound = true
                device = dev
            }
            
        }

        if bFound == false {
            // print("Rayleigh: Device " + deviceName + "Not Found!")
            return 1
        }

        let commandQueue = device.makeCommandQueue()!,
            defaultLibrary = try! device.makeLibrary(filepath: metallib)

        let ForwardSimpleMetalFunction = defaultLibrary.makeFunction(name: "ForwardSimpleMetal")!
        
        let bUseAlignedMemBuffer = UnsafeMutableRawPointer(bUseAlignedMemp)
        let bUseAlignedMem = bUseAlignedMemBuffer.load(as:Int.self)
        
        // if bUseAlignedMem == 1 {
        //     print("Uising aligned memory")
        // }

        let c_wvnb_realBuffer = UnsafeMutableRawPointer(c_wvnb_real)
        let c_wvnb_imagBuffer = UnsafeMutableRawPointer(c_wvnb_imag)
        let MaxDistanceBuffer = UnsafeMutableRawPointer(MaxDistance)
        let mr1Buffer = UnsafeMutableRawPointer(mr1p)
        let mr2Buffer = UnsafeMutableRawPointer(mr2p)

        let u0stepsBuffer = UnsafeMutableRawPointer(u0stepsp)
        
        let r2prBuffer = UnsafeMutableRawPointer(r2pr)
        let r1prBuffer =  UnsafeMutableRawPointer(r1pr)
        let a1prBuffer = UnsafeMutableRawPointer(a1pr)
        let u1_realBuffer = UnsafeMutableRawPointer(u1_real)
        let u1_imagBuffer = UnsafeMutableRawPointer(u1_imag)

        let u0steps = u0stepsBuffer.load(as: Int.self) 
        let mr2 = mr2Buffer.load(as: Int.self)   
        let mr1 = mr1Buffer.load(as: Int.self) 
        let MaxDistance = MaxDistanceBuffer.load(as: Float.self)
        let c_wvnb_real = c_wvnb_realBuffer.load(as: Float.self)
        let c_wvnb_imag = c_wvnb_imagBuffer.load(as: Float.self)

        
        var ll = MemoryLayout<Float>.size*mr1*3
        let PAGE_MAP_SIZE = 16384
        var r1prVectorBuffer:MTLBuffer?
        if bUseAlignedMem == 1
        { 
            if ll % PAGE_MAP_SIZE != 0 {
                ll = (Int(ll/PAGE_MAP_SIZE)+1)*PAGE_MAP_SIZE
            }
            r1prVectorBuffer = device.makeBuffer(bytesNoCopy:r1prBuffer, length: ll, options: [],deallocator:nil)
        }
        else {
            r1prVectorBuffer = device.makeBuffer(bytes: r1prBuffer, length: MemoryLayout<Float>.size*mr1*3, options: [])
        }
       
        var r2prVectorBuffer:MTLBuffer?
        ll = MemoryLayout<Float>.size*mr2*3

        if bUseAlignedMem == 1
        { 
            if ll % PAGE_MAP_SIZE != 0 {
                ll = (Int(ll/PAGE_MAP_SIZE)+1)*PAGE_MAP_SIZE
            }
            r2prVectorBuffer = device.makeBuffer(bytesNoCopy:r2prBuffer, length: ll, options: [],deallocator:nil)
            
        }
        else {
            r2prVectorBuffer = device.makeBuffer(bytes: r2prBuffer, length: ll, options: [])
        }

        var a1prVectorBuffer:MTLBuffer?
        var u1_realVectorBuffer:MTLBuffer?
        var u1_imagVectorBuffer:MTLBuffer?
        var ll2 : Int = 0
        if u0steps>0
        {
            ll2 = MemoryLayout<Float>.size*mr1*mr2
        }
        else {
            ll2 = MemoryLayout<Float>.size*mr1
        }

        ll = MemoryLayout<Float>.size*mr1
            
        if bUseAlignedMem == 1
        { 
            if ll % PAGE_MAP_SIZE != 0 {
                ll = (Int(ll/PAGE_MAP_SIZE)+1)*PAGE_MAP_SIZE
            }
            if ll2 % PAGE_MAP_SIZE != 0 {
                ll2 = (Int(ll2/PAGE_MAP_SIZE)+1)*PAGE_MAP_SIZE
            }
            a1prVectorBuffer = device.makeBuffer(bytesNoCopy:a1prBuffer, length: ll, options: [],deallocator:nil)
            u1_realVectorBuffer = device.makeBuffer(bytesNoCopy:u1_realBuffer, length: ll2, options: [],deallocator:nil)
            u1_imagVectorBuffer = device.makeBuffer(bytesNoCopy:u1_imagBuffer, length: ll2, options: [],deallocator:nil)
            
        }
        else {
            a1prVectorBuffer = device.makeBuffer(bytes: a1prBuffer, length: ll, options: [])
            u1_realVectorBuffer = device.makeBuffer(bytes: u1_realBuffer, length: ll2, options: [])
            u1_imagVectorBuffer = device.makeBuffer(bytes: u1_imagBuffer, length: ll2, options: [])
        }

        ll = MemoryLayout<Float>.size*mr2

        var py_data_u2_realRef = UnsafeMutablePointer(py_data_u2_real)
        var py_data_u2_realVectorBuffer:MTLBuffer?
        var py_data_u2_imagRef = UnsafeMutablePointer(py_data_u2_imag) 
        var py_data_u2_imagVectorBuffer:MTLBuffer?
        if bUseAlignedMem == 1 
        {
            if ll % PAGE_MAP_SIZE != 0 {
                ll = (Int(ll/PAGE_MAP_SIZE)+1)*PAGE_MAP_SIZE
            }
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
    
        let NonBlockingstep = Int(50000e6)
        var nm_step = Int(NonBlockingstep/mr1)


        if nm_step > mr2
        {
            nm_step=mr2
        }
        if nm_step<5
        {
            nm_step=5
        }

       

        var basemr2 = Int(0)
        var n2Limit = Int(0)
        var offset = Int(0)


           
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

            offset = basemr2*u0steps


            let commandBuffer = commandQueue.makeCommandBuffer()!
            let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

            let computePipelineState = try device.makeComputePipelineState(function: ForwardSimpleMetalFunction)
            computeCommandEncoder.setComputePipelineState(computePipelineState)

            var args = RayleighArguments(c_wvnb_real:c_wvnb_real,
                                         c_wvnb_imag:c_wvnb_imag,
                                         MaxDistance:MaxDistance,
                                         mr1step: UInt32(u0steps),
                                         mr1: UInt32(mr1),
                                         mr2:UInt32(n2Limit),
                                         n2BaseSteps:UInt32(offset))

            computeCommandEncoder.setBytes(&args, length: MemoryLayout<RayleighArguments>.stride, index: 0)
            computeCommandEncoder.setBuffer(r2prVectorBuffer, 
                            offset: MemoryLayout<Float>.size*basemr2*3, index: 1)
            computeCommandEncoder.setBuffer(r1prVectorBuffer, offset: 0, index: 2)
            computeCommandEncoder.setBuffer(a1prVectorBuffer, offset: 0, index: 3)
            computeCommandEncoder.setBuffer(u1_realVectorBuffer, offset: 0, index: 4)
            computeCommandEncoder.setBuffer(u1_imagVectorBuffer, offset: 0, index: 5)
            computeCommandEncoder.setBuffer(py_data_u2_realVectorBuffer, offset: MemoryLayout<Float>.size*basemr2, index: 6)
            computeCommandEncoder.setBuffer(py_data_u2_imagVectorBuffer, offset: MemoryLayout<Float>.size*basemr2, index: 7)

            
            let w = computePipelineState.threadExecutionWidth
            let h = computePipelineState.maxTotalThreadsPerThreadgroup / w

            let numThreadgroups = MTLSize(width: (Int(n2Limit)+(w*h-1))/(w*h), height: 1, depth: 1)
            let threadsPerThreadgroup = MTLSize(width: w*h, height: 1, depth: 1)

            computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
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
            r1prVectorBuffer!.setPurgeableState(MTLPurgeableState.empty)
            r2prVectorBuffer!.setPurgeableState(MTLPurgeableState.empty)
            a1prVectorBuffer!.setPurgeableState(MTLPurgeableState.empty)
            u1_realVectorBuffer!.setPurgeableState(MTLPurgeableState.empty)
            u1_imagVectorBuffer!.setPurgeableState(MTLPurgeableState.empty)
            py_data_u2_realVectorBuffer!.setPurgeableState(MTLPurgeableState.empty)
            py_data_u2_imagVectorBuffer!.setPurgeableState(MTLPurgeableState.empty)

        }

        return 0
    } catch {
        print("\(error)")
        return 1
    }
}

