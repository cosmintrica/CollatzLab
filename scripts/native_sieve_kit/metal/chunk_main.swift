import Dispatch
import Darwin
import Foundation
import Metal

/// Packed layout must match ``CollatzChunkPartial`` in ``CollatzLabSieve.metal``.
/// Metal uses C rules: `int32`+`int32`+`int64`+`int32`+`int32` → 24 bytes, align 8.
private struct CollatzChunkPartial {
    var maxSteps: Int32
    var maxStepsSeedIndex: Int32
    var maxExc: Int64
    var maxExcSeedIndex: Int32
    var pad: Int32
}

/// Expected `sizeof(CollatzChunkPartial)` / array element stride on both Metal and Swift.
private let kCollatzChunkPartialStrideBytes = 24

private func assertCollatzChunkPartialMatchesMetalABI(file: StaticString = #file, line: UInt = #line) {
    let s = MemoryLayout<CollatzChunkPartial>.stride
    let z = MemoryLayout<CollatzChunkPartial>.size
    let a = MemoryLayout<CollatzChunkPartial>.alignment
    precondition(
        s == kCollatzChunkPartialStrideBytes,
        "CollatzChunkPartial stride \(s) != \(kCollatzChunkPartialStrideBytes) — Metal buffer layout mismatch (rebuild + check CollatzLabSieve.metal)",
        file: file,
        line: line
    )
    precondition(
        z == kCollatzChunkPartialStrideBytes,
        "CollatzChunkPartial size \(z) != \(kCollatzChunkPartialStrideBytes)",
        file: file,
        line: line
    )
    precondition(a == 8, "CollatzChunkPartial alignment \(a) != 8 (int64 field)", file: file, line: line)
}

/// Shared Metal pipeline for odd-only lab sieve chunks (one-shot or `--stdio` server).
final class MetalSieveChunkEngine {
    private let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    private let queue: MTLCommandQueue
    /// Reused for every chunk (base odd + count are tiny).
    private let bufBase: MTLBuffer
    private let bufCount: MTLBuffer
    /// Grow-only scratch: per-seed steps only (max_exc reduced on GPU per threadgroup).
    private var bufSteps: MTLBuffer?
    private var perSeedCapacity: Int = 0

    private static let threadgroupWidth = 512

    init(executablePath: String) throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "CollatzLabSieve", code: 3, userInfo: [NSLocalizedDescriptionKey: "Metal: no device"])
        }
        device = dev
        let exeURL = URL(fileURLWithPath: executablePath)
        let thisDir = exeURL.deletingLastPathComponent()
        let libURL = thisDir.appendingPathComponent("CollatzLabSieve.metallib")
        guard FileManager.default.fileExists(atPath: libURL.path) else {
            throw NSError(
                domain: "CollatzLabSieve",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "Missing CollatzLabSieve.metallib (run build_metal_sieve_chunk.sh)"]
            )
        }
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(URL: libURL)
        } catch {
            throw NSError(domain: "CollatzLabSieve", code: 5, userInfo: [NSLocalizedDescriptionKey: "load library: \(error)"])
        }
        guard let fn = library.makeFunction(name: "collatz_lab_sieve_odd") else {
            throw NSError(domain: "CollatzLabSieve", code: 6, userInfo: [NSLocalizedDescriptionKey: "kernel collatz_lab_sieve_odd missing"])
        }
        do {
            pipeline = try device.makeComputePipelineState(function: fn)
        } catch {
            throw NSError(domain: "CollatzLabSieve", code: 7, userInfo: [NSLocalizedDescriptionKey: "pipeline: \(error)"])
        }
        assertCollatzChunkPartialMatchesMetalABI()
        guard let q = device.makeCommandQueue() else {
            throw NSError(domain: "CollatzLabSieve", code: 8, userInfo: [NSLocalizedDescriptionKey: "command queue failed"])
        }
        queue = q

        guard let bb = device.makeBuffer(length: MemoryLayout<Int64>.stride, options: .storageModeShared),
              let bc = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        else {
            throw NSError(domain: "CollatzLabSieve", code: 12, userInfo: [NSLocalizedDescriptionKey: "fixed buffer alloc failed"])
        }
        bufBase = bb
        bufCount = bc
    }

    private func ensureStepsCapacity(_ n: Int) throws {
        if n <= perSeedCapacity {
            return
        }
        guard let bs = device.makeBuffer(length: n * MemoryLayout<Int32>.stride, options: .storageModeShared) else {
            throw NSError(domain: "CollatzLabSieve", code: 13, userInfo: [NSLocalizedDescriptionKey: "steps buffer alloc failed"])
        }
        bufSteps = bs
        perSeedCapacity = n
    }

    /// Run one chunk; returns JSON-serializable aggregate dict (incl. overflow_seeds).
    func runChunk(firstOdd: Int64, count: UInt32) throws -> [String: Any] {
        guard count > 0 else {
            return [
                "max_total_stopping_time": ["n": NSNumber(value: firstOdd), "value": NSNumber(value: -1)],
                "max_stopping_time": ["n": NSNumber(value: firstOdd), "value": NSNumber(value: -1)],
                "max_excursion": ["n": NSNumber(value: firstOdd), "value": NSNumber(value: -1)],
                "overflow_seeds": [],
            ]
        }

        let n = Int(count)
        try ensureStepsCapacity(n)
        guard let bufSteps = bufSteps else {
            throw NSError(domain: "CollatzLabSieve", code: 9, userInfo: [NSLocalizedDescriptionKey: "buffer alloc failed"])
        }

        bufBase.contents().bindMemory(to: Int64.self, capacity: 1).pointee = firstOdd
        bufCount.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = count

        let wg = MetalSieveChunkEngine.threadgroupWidth
        let numGroups = (n + wg - 1) / wg
        let partialStride = kCollatzChunkPartialStrideBytes
        guard let bufPartials = device.makeBuffer(length: numGroups * partialStride, options: .storageModeShared)
        else {
            throw NSError(domain: "CollatzLabSieve", code: 14, userInfo: [NSLocalizedDescriptionKey: "partials buffer alloc failed"])
        }

        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw NSError(domain: "CollatzLabSieve", code: 10, userInfo: [NSLocalizedDescriptionKey: "encoder failed"])
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(bufBase, offset: 0, index: 0)
        enc.setBuffer(bufCount, offset: 0, index: 1)
        enc.setBuffer(bufSteps, offset: 0, index: 2)
        enc.setBuffer(bufPartials, offset: 0, index: 3)
        enc.dispatchThreadgroups(
            MTLSize(width: numGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: wg, height: 1, depth: 1)
        )
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let stepsPtr = bufSteps.contents().bindMemory(to: Int32.self, capacity: n)
        let partialPtr = bufPartials.contents().bindMemory(to: CollatzChunkPartial.self, capacity: numGroups)

        var overflowSeeds: [Int64] = []
        var bestTstN = firstOdd
        var bestTstV: Int32 = -1
        var bestExcN = firstOdd
        var bestExcV: Int64 = -1

        for t in 0..<n {
            let seed = firstOdd + 2 * Int64(t)
            let st = stepsPtr[t]
            if st < 0 {
                overflowSeeds.append(seed)
            }
        }

        for g in 0..<numGroups {
            let p = partialPtr[g]
            if p.maxSteps >= 0 && p.maxSteps > bestTstV {
                bestTstV = p.maxSteps
                bestTstN = firstOdd + 2 * Int64(p.maxStepsSeedIndex)
            }
            if p.maxExc >= 0 && p.maxExc > bestExcV {
                bestExcV = p.maxExc
                bestExcN = firstOdd + 2 * Int64(p.maxExcSeedIndex)
            }
        }

        let overflowNums = overflowSeeds.map { NSNumber(value: $0) }
        return [
            "max_total_stopping_time": [
                "n": NSNumber(value: bestTstN),
                "value": NSNumber(value: bestTstV),
            ],
            "max_stopping_time": [
                "n": NSNumber(value: bestTstN),
                "value": NSNumber(value: bestTstV),
            ],
            "max_excursion": [
                "n": NSNumber(value: bestExcN),
                "value": NSNumber(value: bestExcV),
            ],
            "overflow_seeds": overflowNums,
        ]
    }

    func jsonChunk(firstOdd: Int64, count: UInt32) throws -> String {
        let out = try runChunk(firstOdd: firstOdd, count: count)
        guard JSONSerialization.isValidJSONObject(out),
              let data = try? JSONSerialization.data(withJSONObject: out, options: []),
              let s = String(data: data, encoding: .utf8) else {
            throw NSError(domain: "CollatzLabSieve", code: 11, userInfo: [NSLocalizedDescriptionKey: "json encode failed"])
        }
        return s
    }
}

private func writeLine(_ s: String) {
    guard let data = (s + "\n").data(using: .utf8) else { return }
    FileHandle.standardOutput.write(data)
    fflush(stdout)
}

/// When ``COLLATZ_METAL_SIEVE_STDIO_PIPELINE`` is not ``0``/``false``/``no``, a reader thread
/// buffers the next stdin line while the GPU runs the current chunk (overlap).
private func metalStdioUsePipeline() -> Bool {
    let raw = ProcessInfo.processInfo.environment["COLLATZ_METAL_SIEVE_STDIO_PIPELINE"] ?? "1"
    let s = raw.trimmingCharacters(in: .whitespaces).lowercased()
    if s.isEmpty { return true }
    return !(s == "0" || s == "false" || s == "no" || s == "off")
}

private final class StdioLineBuffer {
    private let lock = NSLock()
    private var lines: [String] = []
    private var closed = false
    private let sem = DispatchSemaphore(value: 0)

    func push(_ line: String) {
        lock.lock()
        lines.append(line)
        lock.unlock()
        sem.signal()
    }

    func closeWriter() {
        lock.lock()
        closed = true
        lock.unlock()
        sem.signal()
    }

    /// ``nil`` when stdin closed and queue drained.
    func popBlocking() -> String? {
        while true {
            sem.wait()
            lock.lock()
            if !lines.isEmpty {
                let s = lines.removeFirst()
                lock.unlock()
                return s
            }
            if closed {
                lock.unlock()
                return nil
            }
            lock.unlock()
        }
    }
}

/// Returns ``false`` if the peer sent ``quit`` (caller should stop processing).
private func processStdioLine(_ trimmed: String, engine: MetalSieveChunkEngine) -> Bool {
    guard let d = trimmed.data(using: .utf8),
          let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any]
    else {
        writeLine("{\"error\":\"invalid_json_line\"}")
        return true
    }
    if (obj["op"] as? String)?.lowercased() == "quit" {
        return false
    }
    let fo = (obj["first_odd"] as? NSNumber)?.int64Value ?? 1
    let cntU: UInt32
    if let n = obj["count"] as? NSNumber {
        cntU = n.uint32Value
    } else {
        writeLine("{\"error\":\"missing_count\"}")
        return true
    }
    let firstOdd = fo | 1
    do {
        let js = try engine.jsonChunk(firstOdd: firstOdd, count: cntU)
        writeLine(js)
    } catch {
        writeLine("{\"error\":\"\(String(describing: error))\"}")
    }
    return true
}

func mainChunk() {
    let args = CommandLine.arguments

    // Load pipeline + assert CollatzChunkPartial ABI (same as --stdio / one-shot chunk).
    if args.contains("--ping") {
        do {
            _ = try MetalSieveChunkEngine(executablePath: args[0])
            writeLine("{\"ok\":true,\"stdio\":true,\"metal_abi_ok\":true}")
        } catch {
            fputs("\(error)\n", stderr)
            exit(1)
        }
        return
    }

    if args.contains("--stdio") {
        guard args.count >= 1 else {
            fputs("stdio mode needs executable path in argv[0]\n", stderr)
            exit(2)
        }
        let exePath = args[0]
        let engine: MetalSieveChunkEngine
        do {
            engine = try MetalSieveChunkEngine(executablePath: exePath)
        } catch {
            fputs("\(error)\n", stderr)
            exit(3)
        }
        if metalStdioUsePipeline() {
            let buf = StdioLineBuffer()
            let reader = Thread {
                while let line = readLine(strippingNewline: true) {
                    let tr = line.trimmingCharacters(in: .whitespaces)
                    if tr.isEmpty { continue }
                    buf.push(tr)
                }
                buf.closeWriter()
            }
            reader.name = "collatz-metal-stdio-read"
            reader.start()
            while let trimmed = buf.popBlocking() {
                if !processStdioLine(trimmed, engine: engine) {
                    break
                }
            }
        } else {
            while let line = readLine(strippingNewline: true) {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.isEmpty { continue }
                if !processStdioLine(trimmed, engine: engine) {
                    break
                }
            }
        }
        fflush(stdout)
    }

    var count: UInt32 = 0
    var firstOdd: Int64 = 1
    var i = 1
    while i < args.count {
        let a = args[i]
        if a == "--count", i + 1 < args.count, let v = UInt32(args[i + 1]) {
            count = v
            i += 2
            continue
        }
        if a == "--first-odd", i + 1 < args.count, let v = Int64(args[i + 1]) {
            firstOdd = v | 1
            i += 2
            continue
        }
        i += 1
    }

    if count == 0 {
        fputs(
            "usage: metal_sieve_chunk --ping | --stdio | --first-odd N --count M\n"
                + "  --ping  loads Metal + verifies CollatzChunkPartial layout (exit 1 on failure)\n",
            stderr
        )
        exit(2)
    }

    do {
        let engine = try MetalSieveChunkEngine(executablePath: args[0])
        let js = try engine.jsonChunk(firstOdd: firstOdd, count: count)
        print(js)
    } catch {
        fputs("\(error)\n", stderr)
        exit(12)
    }
}

mainChunk()
