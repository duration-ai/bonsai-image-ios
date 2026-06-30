import Foundation
import MLX
import MLXFast

// Qwen3-4B text encoder — FLUX.2 Klein's prompt conditioning, on-device.
// 36 layers, hidden 2560, GQA 32q/8kv (head_dim 128, per-head q/k RMSNorm),
// SwiGLU MLP (9728), RoPE theta 1e6, rms eps 1e-6. Weights are MLX 4-bit
// (group 64) INCLUDING the embedding table (gather rows + dequantize — never
// expand the 151936×2560 table). Prompt embeds = RAW hidden states after
// layers (9, 18, 27) — HF-style indexing where 0 is the embedding output, so
// list index i+1 = output of 0-indexed layer i — concatenated per token to
// [1, S, 7680]. The final model.norm is NOT applied to taps.
// Spec + goldens: mflux-prism Qwen3TextEncoder / Flux2PromptEncoder
// (fixtures/qwen_fixture.safetensors, made by make_qwen_fixture.py).
public final class Qwen3Encoder {
    public static let maxLen = 512
    public static let padId: Int32 = 151_643
    // Rendered Qwen3 chat template (enable_thinking=false, add_generation_prompt)
    // — pinned by fixture meta; Swift hard-codes the rendering instead of Jinja.
    public static let templatePrefix = "<|im_start|>user\n"
    public static let templateSuffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    static let layers = 36, hid = 2560, nHeads = 32, nKV = 8, headDim = 128
    static let taps: Set<Int> = [9, 18, 27]  // 1-based "list index" = after layer (i-1)
    static let eps: Float = 1e-6

    // var so the 2.1 GB can be FREED before the DiT loads (sequential residency).
    private var W: [String: MLXArray]?

    public init(weightsPath: String) throws {
        W = try loadArrays(url: URL(fileURLWithPath: weightsPath))
    }

    public func evict() {
        W = nil
        GPU.clearCache()
    }

    private func w(_ k: String) -> MLXArray {
        guard let v = W?[k] else { fatalError("qwen3 missing \(k)") }
        return v
    }
    private func qlin(_ key: String, _ a2d: MLXArray) -> MLXArray {
        quantizedMatmul(a2d, w("\(key).weight"), scales: w("\(key).scales"),
                        biases: w("\(key).biases"), transpose: true, groupSize: 64, bits: 4)
    }
    private func rms(_ a: MLXArray, _ key: String) -> MLXArray {
        MLXFast.rmsNorm(a, weight: w(key).asType(a.dtype), eps: Self.eps)
    }
    private func rotateHalf(_ x: MLXArray) -> MLXArray {
        let d = x.shape.last! / 2
        return concatenated([-x[.ellipsis, d...], x[.ellipsis, ..<d]], axis: -1)
    }

    /// tokenIds: the UNPADDED prompt tokens (rendered template, ≤512).
    /// Returns (promptEmbeds [1,512,7680] bf16, textIds [1,512,4] int32).
    public func encode(tokenIds: [Int32]) -> (embeds: MLXArray, textIds: MLXArray) {
        let S = Self.maxLen
        let trueLen = min(tokenIds.count, S)
        var ids = Array(tokenIds.prefix(S))
        ids.append(contentsOf: Array(repeating: Self.padId, count: S - trueLen))

        let idArr = MLXArray(ids).reshaped([1, S])
        // Quantized-embedding gather: rows of packed weight + their group scales.
        let flat = idArr.reshaped([S])
        var h = dequantized(
            w("model.embed_tokens.weight")[flat],
            scales: w("model.embed_tokens.scales")[flat],
            biases: w("model.embed_tokens.biases")[flat],
            groupSize: 64, bits: 4
        ).reshaped([1, S, Self.hid])
        let dt = h.dtype  // bf16 (scales dtype)

        // Additive mask: causal upper-tri -inf + padding columns -inf. [1,1,S,S]
        let idx = MLXArray(0 ..< Int32(S))
        let causal = which(
            idx.reshaped([1, S]) .> idx.reshaped([S, 1]),
            MLXArray(-Float.infinity).asType(dt), MLXArray(Float(0)).asType(dt)
        )
        let padCols = which(
            idx.reshaped([1, S]) .< MLXArray(Int32(trueLen)),
            MLXArray(Float(0)).asType(dt), MLXArray(-Float.infinity).asType(dt)
        )
        let mask = (causal + padCols).reshaped([1, 1, S, S])

        // RoPE tables (fp32 freqs -> dtype), HF layout: emb = [freqs, freqs].
        let inv = 1.0 / pow(MLXArray(Float(1_000_000)), MLXArray(stride(from: 0, to: Self.headDim, by: 2).map { Float($0) / Float(Self.headDim) }))
        let pos = MLXArray(0 ..< Int32(S)).asType(.float32).reshaped([S, 1])
        let freqs = pos * inv.reshaped([1, Self.headDim / 2])
        let emb = concatenated([freqs, freqs], axis: -1)            // [S, headDim]
        let cos = MLX.cos(emb).asType(dt).reshaped([1, 1, S, Self.headDim])
        let sin = MLX.sin(emb).asType(dt).reshaped([1, 1, S, Self.headDim])

        var tapped: [MLXArray] = []
        for i in 0 ..< Self.layers {
            autoreleasepool {
                let p = "model.layers.\(i)"
                let x = rms(h, "\(p).input_layernorm.weight").reshaped([S, Self.hid])
                var q = qlin("\(p).self_attn.q_proj", x).reshaped([1, S, Self.nHeads, Self.headDim])
                var k = qlin("\(p).self_attn.k_proj", x).reshaped([1, S, Self.nKV, Self.headDim])
                let v = qlin("\(p).self_attn.v_proj", x).reshaped([1, S, Self.nKV, Self.headDim])
                    .transposed(0, 2, 1, 3)
                q = rms(q, "\(p).self_attn.q_norm.weight").transposed(0, 2, 1, 3)
                k = rms(k, "\(p).self_attn.k_norm.weight").transposed(0, 2, 1, 3)
                q = q * cos + rotateHalf(q) * sin
                k = k * cos + rotateHalf(k) * sin
                // Fused-GQA sdpa (32q/8kv direct). Verified: manual HF-style
                // repeat_kv gives bit-identical cosines — the residual 3e-4 vs
                // golden is bf16 accumulation noise, not an attention-path diff.
                let attn = MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v,
                    scale: 1.0 / Float(Self.headDim).squareRoot(), mask: mask)
                    .transposed(0, 2, 1, 3).reshaped([S, Self.nHeads * Self.headDim])
                h = h + qlin("\(p).self_attn.o_proj", attn).reshaped([1, S, Self.hid])

                let y = rms(h, "\(p).post_attention_layernorm.weight").reshaped([S, Self.hid])
                let gated = qlin("\(p).mlp.gate_proj", y)
                let mlp = qlin("\(p).mlp.down_proj", (gated * sigmoid(gated)) * qlin("\(p).mlp.up_proj", y))
                h = h + mlp.reshaped([1, S, Self.hid])
                eval(h)
            }
            if Self.taps.contains(i + 1) { tapped.append(h) }
            if (i + 1) % 9 == 0 { bonsaiMemLog("enc:l\(i + 1)/\(Self.layers)") }
        }

        // Per-token concat in tap order == stack(axis:1)+transpose+reshape.
        let embeds = concatenated(tapped, axis: -1)                 // [1, S, 7680]
        // text_ids: (t=0, h=0, w=0, token_index)
        let zeros = MLXArray.zeros([S, 3], dtype: .int32)
        let textIds = concatenated([zeros, MLXArray(0 ..< Int32(S)).reshaped([S, 1])], axis: 1)
            .reshaped([1, S, 4])
        eval(embeds, textIds)
        return (embeds, textIds)
    }
}
