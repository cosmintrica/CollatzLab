import Foundation
import Metal

func runSpike() {
    var count: UInt32 = 500_000
    var baseOdd: UInt64 = 1

    var i = 1
    while i < CommandLine.argc {
        let a = CommandLine.arguments[i]
        if a == "--count", i + 1 < CommandLine.argc, let v = UInt32(CommandLine.arguments[i + 1]) {
            count = v
            i += 2
            continue
        }
        if a == "--base", i + 1 < CommandLine.argc, let v = UInt64(CommandLine.arguments[i + 1]) {
            baseOdd = v | 1
            i += 2
            continue
        }
        i += 1
    }

    guard let device = MTLCreateSystemDefaultDevice() else {
        fputs("Metal: no default device (need Apple GPU or supported Mac).\n", stderr)
        exit(2)
    }

    let thisDir = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
    let libURL = thisDir.appendingPathComponent("CollatzSpike.metallib")
    guard FileManager.default.fileExists(atPath: libURL.path) else {
        fputs("Missing \(libURL.path) — run run.sh to compile the shader.\n", stderr)
        exit(3)
    }

    let library: MTLLibrary
    do {
        library = try device.makeLibrary(URL: libURL)
    } catch {
        fputs("Metal load library: \(error)\n", stderr)
        exit(4)
    }

    guard let fn = library.makeFunction(name: "collatz_odd_descent_spike") else {
        fputs("Kernel collatz_odd_descent_spike not found in metallib.\n", stderr)
        exit(5)
    }

    let pipeline: MTLComputePipelineState
    do {
        pipeline = try device.makeComputePipelineState(function: fn)
    } catch {
        fputs("Pipeline: \(error)\n", stderr)
        exit(6)
    }

    guard let queue = device.makeCommandQueue() else {
        fputs("No command queue.\n", stderr)
        exit(7)
    }

    let stepsBytes = Int(count) * MemoryLayout<UInt32>.stride
    guard let bufBase = device.makeBuffer(length: MemoryLayout<UInt64>.stride, options: .storageModeShared),
          let bufCount = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
          let bufOut = device.makeBuffer(length: stepsBytes, options: .storageModeShared)
    else {
        fputs("Buffer allocation failed.\n", stderr)
        exit(8)
    }

    bufBase.contents().bindMemory(to: UInt64.self, capacity: 1).pointee = baseOdd
    bufCount.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = count

    let wg = 256
    let numGroups = (Int(count) + wg - 1) / wg
    let threadgroups = MTLSize(width: numGroups, height: 1, depth: 1)
    let threadsPerTG = MTLSize(width: wg, height: 1, depth: 1)

    func encode(_ enc: MTLComputeCommandEncoder) {
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(bufBase, offset: 0, index: 0)
        enc.setBuffer(bufCount, offset: 0, index: 1)
        enc.setBuffer(bufOut, offset: 0, index: 2)
        enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
    }

    // Warmup
    if let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() {
        encode(enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    let t0 = CFAbsoluteTimeGetCurrent()
    guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
        fputs("Command buffer failed.\n", stderr)
        exit(9)
    }
    encode(enc)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    let wall = CFAbsoluteTimeGetCurrent() - t0

    let oddsPerSec = Double(count) / max(wall, 1e-9)
    print(String(format: "{\"device\":\"%@\",\"base_odd\":%llu,\"odd_count\":%u,\"wall_s\":%.4f,\"odd_per_sec\":%.0f}",
                 device.name, baseOdd, count, wall, oddsPerSec))
}

runSpike()
