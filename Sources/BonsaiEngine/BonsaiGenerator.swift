import Foundation
import MLX
import MLXRandom

/// Opt-in instrumentation. OFF by default so importing the library is silent;
/// a host app turns it on to capture the per-phase memory trail:
///
///     BonsaiDebug.memLog = true                  // default <Documents>/bonsai-memlog.txt
///     BonsaiDebug.memLogURL = myCustomURL        // optional override
///
/// The log is fsync'd after every line and written to the Documents dir (NOT
/// Caches, which iOS can purge) so it survives an OOM SIGKILL + relaunch — the
/// last line on disk names the phase that blew up. That's the whole point: an
/// OOM is a SIGKILL, and NSLog/os_log are async+buffered, so console output is
/// lost exactly when it matters.
public enum BonsaiDebug {
    // Set-once debug toggles (configure at startup before generating).
    /// Enable the on-disk phase memory log. Default `false`.
    public nonisolated(unsafe) static var memLog = false
    /// Where the log is written when `memLog` is on. Default `<Documents>/bonsai-memlog.txt`.
    public nonisolated(unsafe) static var memLogURL: URL =
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("bonsai-memlog.txt")
}

/// Records MLX footprint + iOS jetsam headroom at a phase boundary (no-op unless
/// `BonsaiDebug.memLog` is on). os_proc_available_memory = bytes the app can
/// still allocate before jetsam.
public func bonsaiMemLog(_ tag: String, reset: Bool = false) {
    guard BonsaiDebug.memLog else { return }
    let s = GPU.snapshot()
    #if os(iOS)
    let availMB = Int(os_proc_available_memory()) / 1_000_000  // jetsam headroom
    #else
    let availMB = -1  // macOS has no jetsam budget; keep the line format stable
    #endif
    let line = "\(tag): active=\(s.activeMemory / 1_000_000)MB  cache=\(s.cacheMemory / 1_000_000)MB  peak=\(s.peakMemory / 1_000_000)MB  headroom=\(availMB)MB\n"
    let url = BonsaiDebug.memLogURL
    let data = Data(line.utf8)
    if reset || !FileManager.default.fileExists(atPath: url.path) {
        try? data.write(to: url, options: .atomic)
    } else if let fh = try? FileHandle(forWritingTo: url) {
        defer { try? fh.close() }
        _ = try? fh.seekToEnd()
        try? fh.write(contentsOf: data)
        try? fh.synchronize()  // fsync — on disk before any subsequent OOM SIGKILL
    }
}

// The full on-device generator: loads the packed transformer + VAE weights, then
// turns a prompt embedding (from the on-device Qwen3 encoder) into an RGB image. Every stage is
// parity-validated (transformer, VAE, RoPE, time-embed, scheduler); the end-to-end
// loop matches the Python pipeline (e2eParity / e2eGenParity).
public final class BonsaiGenerator {
    // var? so the 1.4 GB transformer can be FREED before the VAE decode (evict=true).
    private var transformerW: [String: MLXArray]?   // packed ternary weights (1.4 GB)
    private var model: KleinTransformer?
    private let vaeW: [String: MLXArray]             // small VAE decoder weights + bn stats

    public init(transformerWeightsPath: String, vaeWeightsPath: String) throws {
        let tw = try loadArrays(url: URL(fileURLWithPath: transformerWeightsPath))
        vaeW = try loadArrays(url: URL(fileURLWithPath: vaeWeightsPath))
        transformerW = tw
        model = KleinTransformer(W: tw, B: 1)
    }

    /// promptEmbeds [1,txtSeq,7680] + textIds [txtSeq,4] from the on-device Qwen3 encoder (`Qwen3Encoder.encode`).
    /// `vaeFp32`=false halves VAE memory (bf16, cosine ~0.9986). `evict`=true frees the
    /// transformer (1.4 GB) before the VAE decode — the generator is single-shot after.
    public func generateImageTensor(promptEmbeds: MLXArray, textIds: MLXArray,
                                    height: Int, width: Int, steps: Int, seed: UInt64,
                                    initLatents: MLXArray? = nil, vaeFp32: Bool = true,
                                    evict: Bool = false,
                                    onStep: ((Int, Int) -> Void)? = nil) -> MLXArray {
        let latH = height / 16, latW = width / 16
        let seq = latH * latW
        let sched = FlowMatchScheduler(numSteps: steps, imageSeqLen: (height / 16) * (width / 16))
        // Qwen3Encoder.encode returns textIds as [1, txtSeq, 4]; buildRope works in
        // 2-D [txtSeq, 4] (like latentGridIds and the parity fixtures). Normalize so
        // both the encode()->generate() flow and a pre-squeezed caller work.
        let txtIds2D = textIds.ndim == 3 ? textIds.reshaped([textIds.shape[1], textIds.shape[2]]) : textIds
        let (ropeCos, ropeSin) = buildRope(imgIds: latentGridIds(latH, latW), txtIds: txtIds2D)
        // var? so these LOCAL refs can be released too — niling only the instance
        // properties leaves the 1.4 GB alive via tw/m until the function returns
        // (i.e. past the VAE decode). That was the no-op eviction: active stayed
        // ~1442 MB at "post-evict", so the VAE ran on top of the full transformer.
        var tw: [String: MLXArray]? = transformerW!
        var m: KleinTransformer? = model!
        bonsaiMemLog("loaded, pre-loop")

        var latents = initLatents
            ?? MLXRandom.normal([1, seq, 128], key: MLXRandom.key(seed)).asType(.bfloat16)
        // Pool per step: the step's MTLCommandBuffers (which retain every buffer
        // they touch, incl. the transient ternary-unpack tensors) are autoreleased;
        // draining per iteration releases them instead of accreting all steps.
        for t in 0 ..< steps {
            autoreleasepool {
                let temb = timeGuidanceEmbed(sched.timesteps[t]) { tw![$0]! }.asType(.bfloat16)
                let noise = m!.forward(latents: latents, textEmb: promptEmbeds, temb: temb,
                                       ropeCos: ropeCos, ropeSin: ropeSin)
                latents = sched.step(latents, noise: noise, t: t)
                eval(latents)
            }
            bonsaiMemLog("step \(t + 1)/\(steps)")
            onStep?(t + 1, steps)
        }
        eval(latents)
        // Free the transformer (1.4 GB) before the VAE-decode spike. Release BOTH
        // the instance properties AND the local refs, else the locals retain it.
        if evict {
            model = nil; m = nil
            transformerW = nil; tw = nil
            GPU.clearCache()
            bonsaiMemLog("post-evict, pre-VAE")
            GPU.resetPeakMemory()  // so the vae:* lines report VAE-only peak, not the loop's
        } else {
            GPU.clearCache()
        }
        // decode_packed_latents: pack -> bn-denorm (ε=1e-4) -> unpatchify -> VAE
        let dt: DType = vaeFp32 ? .float32 : .bfloat16
        let packed = latents.reshaped([1, latH, latW, 128]).transposed(0, 3, 1, 2).asType(dt)
        let bnMean = vaeW["bn.running_mean"]!.reshaped([1, 128, 1, 1]).asType(dt)
        let bnStd = sqrt(vaeW["bn.running_var"]!.reshaped([1, 128, 1, 1]).asType(dt) + 1e-4)
        var lat = packed * bnStd + bnMean
        lat = lat.reshaped([1, 32, 2, 2, latH, latW]).transposed(0, 1, 4, 2, 5, 3).reshaped([1, 32, latH * 2, latW * 2])
        // Reset the peak gauge after every logged op so each vae:* line reports that
        // op's OWN transient high-water (not a monotonic max) — names the culprit op.
        let img = vaeDecode(latentNCHW: lat, vw: { self.vaeW[$0]! }, has: { self.vaeW[$0] != nil }, fp32: vaeFp32,
                            log: { bonsaiMemLog("vae:\($0)"); GPU.resetPeakMemory() })
            .asType(.float32)
        eval(img)
        bonsaiMemLog("post-VAE")
        return img
    }

    /// image [1,3,H,W] (model range) -> row-major HWC uint8 RGB (clip(x/2+0.5,0,1)·255).
    public func toRGB8(_ image: MLXArray) -> (rgb: Data, width: Int, height: Int) {
        let h = image.shape[2], w = image.shape[3]
        let hwc = clip(image / 2 + 0.5, min: 0, max: 1).transposed(0, 2, 3, 1).reshaped([h * w * 3])
        let u8 = round(hwc * 255).asType(.uint8)
        eval(u8)
        return (Data(u8.asArray(UInt8.self)), w, h)
    }
}
