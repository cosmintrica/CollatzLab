import Foundation
import Metal

/// Must match ``CollatzChunkPartial`` in ``CollatzLabSieve.metal`` (24 bytes stride, 8-byte align).
private struct CollatzChunkPartial {
    var maxSteps: Int32
    var maxStepsSeedIndex: Int32
    var maxExc: Int64
    var maxExcSeedIndex: Int32
    var pad: Int32
}

private func assertCollatzChunkPartialMetalABI() {
    precondition(MemoryLayout<CollatzChunkPartial>.stride == 24, "CollatzChunkPartial stride != 24")
    precondition(MemoryLayout<CollatzChunkPartial>.size == 24, "CollatzChunkPartial size != 24")
    precondition(MemoryLayout<CollatzChunkPartial>.alignment == 8, "CollatzChunkPartial alignment != 8")
}

/// Run ``collatz_lab_sieve_odd`` and print aggregate JSON (parity with ``sieve_reference`` / C).
func runVerify() {
    var count: UInt32 = 5_000
    var baseOdd: Int64 = 1

    var i = 1
    while i < CommandLine.argc {
        let a = CommandLine.arguments[i]
        if a == "--count", i + 1 < CommandLine.argc, let v = UInt32(CommandLine.arguments[i + 1]) {
            count = v
            i += 2
            continue
        }
        if a == "--base", i + 1 < CommandLine.argc, let v = Int64(CommandLine.arguments[i + 1]) {
            baseOdd = v | 1
            i += 2
            continue
        }
        i += 1
    }

    guard let device = MTLCreateSystemDefaultDevice() else {
        fputs("Metal: no device\n", stderr)
        exit(2)
    }

    let thisDir = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
    let libURL = thisDir.appendingPathComponent("CollatzLabSieve.metallib")
    guard FileManager.default.fileExists(atPath: libURL.path) else {
        fputs("Missing \(libURL.path) — run build_metal_verify.sh\n", stderr)
        exit(3)
    }

    let library: MTLLibrary
    do {
        library = try device.makeLibrary(URL: libURL)
    } catch {
        fputs("load library: \(error)\n", stderr)
        exit(4)
    }

    guard let fn = library.makeFunction(name: "collatz_lab_sieve_odd") else {
        fputs("kernel collatz_lab_sieve_odd missing\n", stderr)
        exit(5)
    }

    let pipeline: MTLComputePipelineState
    do {
        pipeline = try device.makeComputePipelineState(function: fn)
    } catch {
        fputs("pipeline: \(error)\n", stderr)
        exit(6)
    }

    assertCollatzChunkPartialMetalABI()

    let n = Int(count)
    let wg = 512
    let numGroups = (n + wg - 1) / wg
    let partialStride = MemoryLayout<CollatzChunkPartial>.stride

    guard let queue = device.makeCommandQueue(),
          let bufBase = device.makeBuffer(length: MemoryLayout<Int64>.stride, options: .storageModeShared),
          let bufCount = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
          let bufSteps = device.makeBuffer(length: n * MemoryLayout<Int32>.stride, options: .storageModeShared),
          let bufPartials = device.makeBuffer(length: numGroups * partialStride, options: .storageModeShared)
    else {
        fputs("buffer alloc failed\n", stderr)
        exit(7)
    }

    bufBase.contents().bindMemory(to: Int64.self, capacity: 1).pointee = baseOdd
    bufCount.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = count

    guard let cb = queue.makeCommandBuffer(),
          let enc = cb.makeComputeCommandEncoder() else {
        exit(8)
    }
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(bufBase, offset: 0, index: 0)
    enc.setBuffer(bufCount, offset: 0, index: 1)
    enc.setBuffer(bufSteps, offset: 0, index: 2)
    enc.setBuffer(bufPartials, offset: 0, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: wg, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    let partialPtr = bufPartials.contents().bindMemory(to: CollatzChunkPartial.self, capacity: numGroups)

    var bestTstN = baseOdd
    var bestTstV: Int32 = -1
    var bestExcN = baseOdd
    var bestExcV: Int64 = -1

    for g in 0..<numGroups {
        let p = partialPtr[g]
        if p.maxSteps >= 0 && p.maxSteps > bestTstV {
            bestTstV = p.maxSteps
            bestTstN = baseOdd + 2 * Int64(p.maxStepsSeedIndex)
        }
        if p.maxExc >= 0 && p.maxExc > bestExcV {
            bestExcV = p.maxExc
            bestExcN = baseOdd + 2 * Int64(p.maxExcSeedIndex)
        }
    }

    let lastLinear = baseOdd + 2 * (Int64(count) - 1)
    let json = String(
        format: "{\"processed\":%u,\"last_processed\":%lld,\"max_total_stopping_time\":{\"n\":%lld,\"value\":%d},\"max_stopping_time\":{\"n\":%lld,\"value\":%d},\"max_excursion\":{\"n\":%lld,\"value\":%lld}}",
        count,
        lastLinear,
        bestTstN,
        bestTstV,
        bestTstN,
        bestTstV,
        bestExcN,
        bestExcV
    )
    print(json)
}

runVerify()
