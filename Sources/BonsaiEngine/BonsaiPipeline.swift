import Foundation
import MLX

/// One-call, sequential-residency wrapper over a downloaded Bonsai model directory.
///
/// It resolves the four engine inputs from a single `modelDir` so callers don't
/// wire paths by hand, and runs the full prompt→pixels flow with each model
/// **loaded, used, and evicted** before the next — so peak memory is the largest
/// single phase (~2.3 GB encoder) rather than the sum of all three.
///
/// Expected layout under `modelDir` (the `prism-ml/bonsai-image-ternary-4B-mlx-2bit`
/// download, with the VAE artifact placed alongside it):
/// ```
///   transformer-packed-mflux/diffusion_pytorch_model.safetensors   DiT (ternary 2-bit)
///   text_encoder-mlx-4bit/model.safetensors                        Qwen3-4B encoder (4-bit g64)
///   tokenizer/                                                     tokenizer.json + config
///   vae.safetensors                                                small-decoder artifact
/// ```
public final class BonsaiPipeline {
    /// Resolved absolute paths to the four inputs. Build from a `modelDir` root,
    /// or set each path explicitly (e.g. when files live in separate bundles).
    public struct Paths {
        public var encoder: String
        public var tokenizer: String
        public var transformer: String
        public var vae: String

        /// Derive the standard sub-paths from a single downloaded model directory.
        /// `vaePath` overrides the default `<modelDir>/vae.safetensors`.
        public init(modelDir: String, vaePath: String? = nil) {
            encoder = "\(modelDir)/text_encoder-mlx-4bit/model.safetensors"
            tokenizer = "\(modelDir)/tokenizer"
            transformer = "\(modelDir)/transformer-packed-mflux/diffusion_pytorch_model.safetensors"
            vae = vaePath ?? "\(modelDir)/vae.safetensors"
        }

        public init(encoder: String, tokenizer: String, transformer: String, vae: String) {
            self.encoder = encoder; self.tokenizer = tokenizer
            self.transformer = transformer; self.vae = vae
        }
    }

    /// Coarse pipeline stage, reported to `progress` alongside a 0...1 fraction.
    public enum Phase: String, Sendable { case encoding, denoising, decoding, done }

    /// Every file the pipeline will read, with a human label — handy for a
    /// pre-flight "are the weights present?" check in a host app.
    public var requiredFiles: [(label: String, path: String)] {
        [("Qwen3 encoder", paths.encoder), ("tokenizer", paths.tokenizer),
         ("DiT transformer", paths.transformer), ("VAE", paths.vae)]
    }

    private let paths: Paths

    public init(modelDir: String, vaePath: String? = nil) {
        self.paths = Paths(modelDir: modelDir, vaePath: vaePath)
    }
    public init(paths: Paths) { self.paths = paths }

    /// True only if all four inputs exist on disk.
    public func weightsPresent() -> Bool {
        let fm = FileManager.default
        return requiredFiles.allSatisfy { fm.fileExists(atPath: $0.path) }
    }

    /// Full prompt→RGB generation, single-shot. Runs **synchronously** on the
    /// calling thread (heavy + blocking — call from a background queue, never the
    /// main thread). `progress` is invoked on that same thread at each phase
    /// boundary and once per denoise step, with a monotonic 0...1 fraction.
    ///
    /// Returns row-major HWC uint8 RGB plus its dimensions (feed to `toRGB8`'s
    /// consumers, e.g. a `CGImage`).
    public func generate(prompt: String,
                         height: Int = 512, width: Int = 512,
                         steps: Int = 4, seed: UInt64 = 0,
                         vaeFp32: Bool = false,
                         progress: ((Phase, Double) -> Void)? = nil)
        throws -> (rgb: Data, width: Int, height: Int)
    {
        // 1. Encode the prompt, then free the ~2.3 GB encoder before the DiT loads.
        progress?(.encoding, 0)
        let enc = try Qwen3Encoder(weightsPath: paths.encoder)
        let tok = try QwenTokenizer.load(dir: URL(fileURLWithPath: paths.tokenizer))
        let ids = QwenTokenizer.promptIds(tok, prompt: prompt)
        let (embeds, textIds) = enc.encode(tokenIds: ids)   // materialized inside encode()
        enc.evict()
        progress?(.denoising, 0.10)

        // 2. Denoise (the 1.4 GB transformer is evicted before the VAE decode).
        let gen = try BonsaiGenerator(transformerWeightsPath: paths.transformer,
                                      vaeWeightsPath: paths.vae)
        let image = gen.generateImageTensor(
            promptEmbeds: embeds, textIds: textIds,
            height: height, width: width, steps: steps, seed: seed,
            vaeFp32: vaeFp32, evict: true,
            onStep: { step, total in
                progress?(.denoising, 0.10 + 0.75 * Double(step) / Double(total))
            })
        progress?(.decoding, 0.90)

        // 3. To row-major HWC uint8 RGB.
        let out = gen.toRGB8(image)
        progress?(.done, 1.0)
        return out
    }
}

public extension BonsaiPipeline {
    /// MLX GPU footprint plus (iOS) the memory still allocatable before jetsam,
    /// all in MB. Cheap — safe to poll from a `progress` callback to drive a live
    /// memory readout. `availableMB` is -1 off-iOS (no per-app jetsam budget).
    static func memorySnapshotMB() -> (activeMB: Int, cacheMB: Int, availableMB: Int) {
        let s = GPU.snapshot()
        #if os(iOS)
        let avail = Int(os_proc_available_memory()) / 1_000_000
        #else
        let avail = -1
        #endif
        return (Int(s.activeMemory) / 1_000_000, Int(s.cacheMemory) / 1_000_000, avail)
    }
}
