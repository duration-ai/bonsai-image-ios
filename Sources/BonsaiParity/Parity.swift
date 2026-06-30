import Foundation
import MLX
import MLXFast
import BonsaiEngine

// macOS parity harnesses for each ported component, validating BonsaiEngine
// against Python ground-truth fixtures (scripts/make_*_fixture.py). Run via
// `BonsaiParity {transformer|vae|preproc|e2e|e2egen}`.

// Paths are env-overridable so the harness runs from any checkout (see README):
//   BONSAI_MODEL     — the downloaded bonsai-image-4B-ternary-mlx model directory
//   BONSAI_FIXTURES  — the golden-fixture directory (made by scripts/make_*_fixture.py)
private let MODEL_DIR = ProcessInfo.processInfo.environment["BONSAI_MODEL"]
    ?? "\(NSHomeDirectory())/models/bonsai-image-4B-ternary-mlx"
private let FIX = ProcessInfo.processInfo.environment["BONSAI_FIXTURES"] ?? "fixtures"
private let MODEL = "\(MODEL_DIR)/transformer-packed-mflux/diffusion_pytorch_model.safetensors"

func transformerParity() throws {
    let W = try loadArrays(url: URL(fileURLWithPath: MODEL))
    let fx = try loadArrays(url: URL(fileURLWithPath: "\(FIX)/full_transformer_fixture.safetensors"))
    func fxa(_ k: String) -> MLXArray { fx[k]! }
    let latents = fxa("latents"), target = fxa("output").asType(.float32)
    let model = KleinTransformer(W: W, B: latents.shape[0])
    let out = model.forward(latents: latents, textEmb: fxa("text_embeddings"), temb: fxa("temb"),
                            ropeCos: fxa("rope_cos"), ropeSin: fxa("rope_sin")).asType(.float32)
    eval(out, target)
    let maxd = (out - target).abs().max().item(Float.self)
    let cos = (out * target).sum().item(Float.self)
        / ((out * out).sum().sqrt().item(Float.self) * (target * target).sum().sqrt().item(Float.self))
    print(String(format: "maxAbsDiff=%.4f mean|t|=%.4f cosine=%.8f", maxd, target.abs().mean().item(Float.self), cos))
    print(cos > 0.999 ? "✅ FULL 25-BLOCK TRANSFORMER PARITY PASS (via KleinTransformer struct)" : "❌ PARITY FAIL")
}

func vaeParity() throws {
    let fp32 = ProcessInfo.processInfo.environment["BONSAI_FP32"] == "1"
    let V = try loadArrays(url: URL(fileURLWithPath: "\(FIX)/vae_fixture.safetensors"))
    func vw(_ k: String) -> MLXArray { V[k]! }
    func has(_ k: String) -> Bool { V[k] != nil }
    let out = vaeDecode(latentNCHW: vw("latent"), vw: vw, has: has, fp32: fp32).asType(.float32)
    let target = vw(fp32 ? "image_f32" : "image").asType(.float32)
    eval(out, target)
    let maxd = (out - target).abs().max().item(Float.self)
    let cos = (out * target).sum().item(Float.self)
        / ((out * out).sum().sqrt().item(Float.self) * (target * target).sum().sqrt().item(Float.self))
    print(String(format: "[%@] out %@  maxAbsDiff=%.4f mean|t|=%.4f cosine=%.8f",
                 fp32 ? "fp32" : "bf16", String(describing: out.shape), maxd, target.abs().mean().item(Float.self), cos))
    if fp32 {
        print(cos > 0.99995 ? "✅ VAE DECODER PARITY PASS (fp32 golden — conventions proven)" : "❌ PARITY FAIL")
    } else {
        print(cos > 0.997 ? "✅ VAE bf16 accumulation check OK (run fp32 on-device for quality)"
                          : "❌ bf16 cosine unexpectedly low — investigate")
    }
}

func preprocParity() throws {
    let W = try loadArrays(url: URL(fileURLWithPath: MODEL))
    let fx = try loadArrays(url: URL(fileURLWithPath: "\(FIX)/preproc_fixture.safetensors"))
    func w(_ k: String) -> MLXArray { W[k]! }
    func fxa(_ k: String) -> MLXArray { fx[k]! }
    let meta = fxa("meta").asArray(Int32.self)
    let H = Int(meta[0]), Wd = Int(meta[1]), txtSeq = Int(meta[2]), timestep = Float(meta[3])
    let imgIds = latentGridIds(H, Wd)
    let (ropeCos, ropeSin) = buildRope(imgIds: imgIds, txtIds: MLXArray.zeros([txtSeq, 4], type: Int32.self))
    let temb = timeGuidanceEmbed(timestep, w)
    func report(_ name: String, _ out: MLXArray, _ target: MLXArray) -> Bool {
        let o = out.asType(.float32), t = target.asType(.float32)
        eval(o, t)
        let maxd = (o - t).abs().max().item(Float.self)
        let cos = (o * t).sum().item(Float.self)
            / ((o * o).sum().sqrt().item(Float.self) * (t * t).sum().sqrt().item(Float.self))
        print(String(format: "  %-9s maxAbsDiff=%.6f cosine=%.8f", (name as NSString).utf8String!, maxd, cos))
        return cos > 0.99999 && maxd < 1e-3
    }
    let idsMatch = (imgIds .== fxa("img_ids")).all().item(Bool.self)
    print("  grid_ids  exact_match=\(idsMatch)")
    let ok = report("rope_cos", ropeCos, fxa("rope_cos")) && report("rope_sin", ropeSin, fxa("rope_sin"))
        && report("temb", temb, fxa("temb"))
    print(idsMatch && ok ? "✅ PREPROC PARITY PASS (pos_embed + grid_ids + time_embed)" : "❌ PREPROC PARITY FAIL")
}

func e2eParity() throws {
    let W = try loadArrays(url: URL(fileURLWithPath: MODEL))
    let E = try loadArrays(url: URL(fileURLWithPath: "\(FIX)/e2e_fixture.safetensors"))
    let V = try loadArrays(url: URL(fileURLWithPath: "\(FIX)/vae_fixture.safetensors"))
    func w(_ k: String) -> MLXArray { W[k]! }
    func e(_ k: String) -> MLXArray { E[k]! }
    func vv(_ k: String) -> MLXArray { V[k]! }
    func hasV(_ k: String) -> Bool { V[k] != nil }
    let meta = e("meta").asArray(Int32.self)
    let latH = Int(meta[0]), latW = Int(meta[1]), steps = Int(meta[3])
    let sigmas = e("sigmas").asArray(Float.self)
    let timesteps = e("timesteps"), promptEmbeds = e("prompt_embeds")
    let model = KleinTransformer(W: W, B: 1)
    let (ropeCos, ropeSin) = buildRope(imgIds: e("latent_ids"), txtIds: e("text_ids"))
    var latents = e("init_latents")
    for t in 0 ..< steps {
        let temb = timeGuidanceEmbed(timesteps[t].item(Float.self), w).asType(.bfloat16)
        let noise = model.forward(latents: latents, textEmb: promptEmbeds, temb: temb, ropeCos: ropeCos, ropeSin: ropeSin)
        latents = latents + (sigmas[t + 1] - sigmas[t]) * noise
    }
    let packed = latents.reshaped([1, latH, latW, 128]).transposed(0, 3, 1, 2).asType(.float32)
    let bnStd = sqrt(vv("bn.running_var").reshaped([1, 128, 1, 1]).asType(.float32) + 1e-4)
    var lat = packed * bnStd + vv("bn.running_mean").reshaped([1, 128, 1, 1]).asType(.float32)
    lat = lat.reshaped([1, 32, 2, 2, latH, latW]).transposed(0, 1, 4, 2, 5, 3).reshaped([1, 32, latH * 2, latW * 2])
    let image = vaeDecode(latentNCHW: lat, vw: vv, has: hasV, fp32: true).asType(.float32)
    func report(_ name: String, _ out: MLXArray, _ target: MLXArray, _ bar: Float) -> Bool {
        let o = out.asType(.float32), t = target.asType(.float32)
        eval(o, t)
        let maxd = (o - t).abs().max().item(Float.self)
        let cos = (o * t).sum().item(Float.self)
            / ((o * o).sum().sqrt().item(Float.self) * (t * t).sum().sqrt().item(Float.self))
        print(String(format: "  %-14s %@  maxAbsDiff=%.4f cosine=%.8f", (name as NSString).utf8String!,
                     String(describing: o.shape), maxd, cos))
        return cos > bar
    }
    print("  steps=\(steps) sigmas=\(sigmas)")
    let okLat = report("final_latents", latents, e("final_latents"), 0.999)
    let okImg = report("image (fp32)", image, e("image_f32"), 0.9995)
    _ = report("image (vs bf16)", image, e("image"), 0.0)
    print(okLat && okImg ? "✅ END-TO-END PARITY PASS — full generate path runs on mlx-swift" : "❌ END-TO-END PARITY FAIL")
}

func e2eGenParity() throws {
    let E = try loadArrays(url: URL(fileURLWithPath: "\(FIX)/e2e_fixture.safetensors"))
    func e(_ k: String) -> MLXArray { E[k]! }
    let meta = e("meta").asArray(Int32.self)
    let steps = Int(meta[3]), height = Int(meta[4]), width = Int(meta[5])
    let sched = FlowMatchScheduler(numSteps: steps, imageSeqLen: (height / 16) * (width / 16))
    let sigErr = zip(sched.sigmas, e("sigmas").asArray(Float.self)).map { abs($0 - $1) }.max() ?? 9
    print(String(format: "  scheduler sigmas maxErr=%.6f  %@", sigErr, "\(sched.sigmas.map { round($0 * 1e4) / 1e4 })"))
    let gen = try BonsaiGenerator(transformerWeightsPath: MODEL, vaeWeightsPath: "\(FIX)/vae_fixture.safetensors")
    let image = gen.generateImageTensor(promptEmbeds: e("prompt_embeds"), textIds: e("text_ids"),
                                        height: height, width: width, steps: steps, seed: 0,
                                        initLatents: e("init_latents")).asType(.float32)
    let target = e("image_f32").asType(.float32)
    eval(image, target)
    let maxd = (image - target).abs().max().item(Float.self)
    let cos = (image * target).sum().item(Float.self)
        / ((image * image).sum().sqrt().item(Float.self) * (target * target).sum().sqrt().item(Float.self))
    let (rgb, w, h) = gen.toRGB8(image)
    print(String(format: "  BonsaiGenerator image %@  maxAbsDiff=%.4f cosine=%.8f  rgb=%dx%d (%d bytes)",
                 String(describing: image.shape), maxd, cos, w, h, rgb.count))
    print(sigErr < 1e-4 && cos > 0.9995
          ? "✅ BonsaiGenerator PARITY PASS — generator API + ported scheduler match the golden"
          : "❌ BonsaiGenerator PARITY FAIL")
}

// On-device text-encoder parity: tokenizer ids must match HF exactly; the
// Qwen3 forward's tap-concat embeds must match the Python pipeline's
// prompt_embeds. Fixture: make_qwen_fixture.py (3 prompts + meta JSON).
func qwenParity() throws {
    let TE = "\(MODEL_DIR)/text_encoder-mlx-4bit/model.safetensors"
    let TOK = "\(MODEL_DIR)/tokenizer"
    let fx = try loadArrays(url: URL(fileURLWithPath: "\(FIX)/qwen_fixture.safetensors"))
    let meta = try JSONSerialization.jsonObject(
        with: Data(contentsOf: URL(fileURLWithPath: "\(FIX)/qwen_fixture_meta.json"))) as! [String: Any]
    let prompts = meta["prompts"] as! [String]

    // 1. Tokenizer: rendered-template encode must reproduce HF ids exactly.
    let tok = try QwenTokenizer.load(dir: URL(fileURLWithPath: TOK))
    var allIds: [[Int32]] = []
    for (i, p) in prompts.enumerated() {
        let trueLen = fx["p\(i)_attention_mask"]![0].sum().item(Int32.self)
        let gold: [Int32] = fx["p\(i)_input_ids"]![0][0 ..< Int(trueLen)].asArray(Int32.self)
        let ids = QwenTokenizer.promptIds(tok, prompt: p)
        guard ids == gold else {
            print("❌ TOKENIZER MISMATCH p\(i): swift \(ids.prefix(8)) vs gold \(gold.prefix(8))")
            exit(1)
        }
        print("p\(i): tokenizer ids exact (\(trueLen) tokens)")
        allIds.append(ids)
    }

    // 2. Encoder forward vs golden embeds.
    let enc = try Qwen3Encoder(weightsPath: TE)
    var worst: Float = 1.0
    for i in prompts.indices {
        let (pe, ti) = enc.encode(tokenIds: allIds[i])
        let out = pe.asType(.float32)
        let target = fx["p\(i)_prompt_embeds"]!.asType(.float32)
        eval(out, target)
        let maxd = (out - target).abs().max().item(Float.self)
        let cos = (out * target).sum().item(Float.self)
            / ((out * out).sum().sqrt().item(Float.self) * (target * target).sum().sqrt().item(Float.self))
        worst = min(worst, cos)
        // text_ids: zeros in t/h/w, arange in the token column (spec-trivial).
        let lastCol = ti[0, 0..., 3].asArray(Int32.self)
        precondition(lastCol == Array(0 ..< Int32(512)) , "text_ids arange wrong")
        print(String(format: "p%d: embeds maxAbsDiff=%.4f cosine=%.8f", i, maxd, cos))
    }
    print(worst > 0.999 ? "✅ QWEN3 ENCODER PARITY PASS (worst cosine \(worst))"
                        : "❌ QWEN PARITY FAIL (worst cosine \(worst))")
}

// Full end-to-end run via the BonsaiPipeline convenience API against the real
// downloaded model dir (BONSAI_MODEL) — the same code path a host app takes.
// Writes a PPM (convert with `sips -s format png`). Env overrides:
// BONSAI_PROMPT / BONSAI_SIZE / BONSAI_STEPS / BONSAI_OUT.
func pipelineGen() throws {
    let env = ProcessInfo.processInfo.environment
    let prompt = env["BONSAI_PROMPT"]
        ?? "a meticulously pruned bonsai tree in a ceramic pot, soft studio lighting, photorealistic"
    let size = Int(env["BONSAI_SIZE"] ?? "512") ?? 512
    let steps = Int(env["BONSAI_STEPS"] ?? "4") ?? 4
    let out = env["BONSAI_OUT"] ?? "/tmp/bonsai-out.ppm"

    let pipe = BonsaiPipeline(modelDir: MODEL_DIR)
    print("model dir: \(MODEL_DIR)")
    for f in pipe.requiredFiles {
        print("  [\(FileManager.default.fileExists(atPath: f.path) ? "ok" : "MISSING")] \(f.label): \(f.path)")
    }
    guard pipe.weightsPresent() else { print("❌ weights missing — see paths above"); exit(1) }

    print("prompt: \(prompt)\nsize: \(size)  steps: \(steps)")
    let t0 = Date()
    let (rgb, w, h) = try pipe.generate(
        prompt: prompt, height: size, width: size, steps: steps, seed: 0,
        progress: { phase, frac in
            let m = BonsaiPipeline.memorySnapshotMB()
            print(String(format: "  %-9@ %3.0f%%  active=%dMB", phase.rawValue as NSString, frac * 100, m.activeMB))
        })
    let secs = Date().timeIntervalSince(t0)

    let bytes = [UInt8](rgb)
    let mean = bytes.reduce(0.0) { $0 + Double($1) } / Double(max(1, bytes.count))
    let mn = bytes.min() ?? 0, mx = bytes.max() ?? 0
    print(String(format: "image %dx%d  %.1fs  pixels: mean=%.1f min=%d max=%d", w, h, secs, mean, mn, mx))

    var ppm = Data("P6\n\(w) \(h)\n255\n".utf8)
    ppm.append(rgb)
    try ppm.write(to: URL(fileURLWithPath: out))
    print("wrote \(out)")
    print(mn != mx && mean > 5 && mean < 250
          ? "✅ PIPELINE E2E OK — non-degenerate image rendered via BonsaiPipeline"
          : "⚠️ image looks degenerate — inspect")
}
