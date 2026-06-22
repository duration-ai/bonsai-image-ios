import Foundation
import MLX
import MLXFast

// Flux2 VAE decoder (vae_variant="small": block_out_channels=(96,192,384,384)).
// decode(): post_quant_conv -> conv_in -> mid(resnet,attn,resnet) -> 4 up-blocks
// (3 resnets, +nearest-2x upsample except last) -> conv_norm_out -> silu -> conv_out.
// Works NHWC throughout (transpose only at the two ends); pytorch-compatible GroupNorm
// in fp32 over grouped channels; single-head spatial self-attention; conv2d weight
// layout [out,kh,kw,in] matches the fixture directly. 16x16 latent -> 128x128 image.

// Reusable: latent NCHW [B,32,h,w] -> image NCHW [B,3,8h,8w]. vw = weight accessor,
// has = key-presence, fp32 selects the (one-shot, high-quality) fp32 path.
// `log`: optional per-stage hook (called AFTER each stage is eval'd) — see below.
public func vaeDecode(latentNCHW: MLXArray, vw: (String) -> MLXArray, has: (String) -> Bool, fp32: Bool,
                      log: ((String) -> Void)? = nil) -> MLXArray {
    let GROUPS = 32
    let GN_EPS: Float = 1e-6
    let WD: DType = fp32 ? .float32 : .bfloat16
    func wt(_ k: String) -> MLXArray { vw(k).asType(WD) }
    func silu(_ a: MLXArray) -> MLXArray { a * sigmoid(a) }
    func conv(_ x: MLXArray, _ key: String, _ pad: Int) -> MLXArray {
        let w = wt("\(key).weight")  // [out, kh, kw, in]
        // A14 GPU page fault (BIF0): MLX's Metal conv kernel faults at 256²-spatial
        // shapes (reproduced on BOTH 0.25.4 and 0.31.4; dies at up3 conv2 96ch@256²
        // with ~3.5GB free — kernel bug, not memory). Dodge the conv kernel for
        // large spatial: explicit im2col + matmul — gemm is the most battle-tested
        // path in MLX (the whole transformer runs on it). Same math; ~113-226MB
        // patches transient at 256², well within budget. BONSAI_GEMM_CONV forces
        // this path everywhere so the Mac parity goldens can exercise it.
        if x.shape[1] * x.shape[2] >= 256 * 256
            || ProcessInfo.processInfo.environment["BONSAI_GEMM_CONV"] != nil {
            let b = x.shape[0], h = x.shape[1], wd = x.shape[2], cin = x.shape[3]
            let kh = w.shape[1], kw = w.shape[2], cout = w.shape[0]
            let xp = pad > 0
                ? padded(x, widths: [IntOrPair(0), IntOrPair(pad), IntOrPair(pad), IntOrPair(0)])
                : x
            var cols: [MLXArray] = []
            for dy in 0 ..< kh { for dx in 0 ..< kw {
                cols.append(xp[0..., dy ..< (dy + h), dx ..< (dx + wd), 0...])
            } }
            // patch order (kh, kw, cin) matches w.transposed(1,2,3,0) -> [kh,kw,in,out]
            let patches = concatenated(cols, axis: 3).reshaped([b * h * wd, kh * kw * cin])
            let wm = w.transposed(1, 2, 3, 0).reshaped([kh * kw * cin, cout])
            return (matmul(patches, wm) + wt("\(key).bias")).reshaped([b, h, wd, cout])
        }
        return conv2d(x, w, stride: 1, padding: IntOrPair(pad)) + wt("\(key).bias")
    }
    func gnorm(_ x: MLXArray, _ key: String) -> MLXArray {  // pytorch-compatible GN: fp32 stats, affine, -> WD
        // Reduction-based (no transposed copies): the old reshape->transpose->layerNorm
        // ->transpose->reshape dance materialized ~6 full-size fp32 tensors per call and
        // was the dominant term in the measured 1117MB/resnet peak at 128² on-device
        // (×4 at 256² = the up3 OOM). [b, h*w, G, gs] is a contiguous VIEW of NHWC
        // (channels innermost), so stats over axes (1,3) == GN over (spatial, group).
        let b = x.shape[0], hh = x.shape[1], ww = x.shape[2], c = x.shape[3]
        let gs = c / GROUPS
        let xf = x.reshaped([b, hh * ww, GROUPS, gs]).asType(.float32)
        let mean = xf.mean(axes: [1, 3], keepDims: true)                      // [b,1,G,1]
        let inv = rsqrt(xf.variance(axes: [1, 3], keepDims: true) + GN_EPS)   // [b,1,G,1]
        let norm = ((xf - mean) * inv).reshaped([b, hh, ww, c])
        return (vw("\(key).weight").asType(.float32) * norm + vw("\(key).bias").asType(.float32)).asType(WD)
    }
    func vlin(_ x: MLXArray, _ key: String) -> MLXArray { matmul(x, wt("\(key).weight").transposed()) + wt("\(key).bias") }
    // Materialize + (optionally) log at op granularity. The lean-gnorm rewrite moved
    // NO peak (1117MB at up2.r0 before and after, to the digit) — so the peak is one
    // op's own transient, not graph accumulation. Per-op eval + per-op peak (the log
    // hook resets the peak gauge) names the culprit op on-device.
    func ev(_ a: MLXArray) -> MLXArray { autoreleasepool { eval(a) }; return a }
    func ck(_ name: String, _ a: MLXArray) -> MLXArray { autoreleasepool { eval(a) }; log?(name); return a }
    func resnet(_ x: MLXArray, _ p: String, _ tag: String) -> MLXArray {  // NHWC in/out; per-op trace
        var h = ck("\(tag).gn1", gnorm(x, "\(p).norm1"))
        h = ck("\(tag).conv1", conv(silu(h), "\(p).conv1", 1))
        h = ck("\(tag).gn2", gnorm(h, "\(p).norm2"))
        h = ck("\(tag).conv2", conv(silu(h), "\(p).conv2", 1))
        let res = has("\(p).conv_shortcut.weight") ? ck("\(tag).sc", conv(x, "\(p).conv_shortcut", 0)) : x
        return ck("\(tag).add", h + res)
    }
    func upsample(_ x: MLXArray, _ p: String) -> MLXArray {  // nearest 2x (NHWC: H=1,W=2) + conv
        conv(repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2), "\(p).conv", 1)
    }
    func attention(_ x: MLXArray, _ p: String) -> MLXArray {  // single-head spatial self-attn
        let b = x.shape[0], hh = x.shape[1], ww = x.shape[2], c = x.shape[3]
        let n = gnorm(x, "\(p).group_norm")
        func head(_ key: String) -> MLXArray { vlin(n, key).reshaped([b, hh * ww, 1, c]).transposed(0, 2, 1, 3) }
        let a = MLXFast.scaledDotProductAttention(
            queries: head("\(p).to_q"), keys: head("\(p).to_k"), values: head("\(p).to_v"),
            scale: 1.0 / Float(c).squareRoot(), mask: nil)
        return x + vlin(a.transposed(0, 2, 1, 3).reshaped([b, hh, ww, c]), "\(p).to_out")
    }

    var x = latentNCHW.asType(WD).transposed(0, 2, 3, 1)   // NHWC
    x = conv(x, "post_quant_conv", 0)
    x = ck("conv_in", conv(x, "decoder.conv_in", 1))
    x = resnet(x, "decoder.mid_block.resnets.0", "mid.r0")
    x = ck("mid.attn", attention(x, "decoder.mid_block.attentions.0"))
    x = resnet(x, "decoder.mid_block.resnets.1", "mid.r1")
    for ub in 0 ..< 4 {
        for r in 0 ..< 3 { x = resnet(x, "decoder.up_blocks.\(ub).resnets.\(r)", "up\(ub).r\(r)") }
        if has("decoder.up_blocks.\(ub).upsamplers.0.conv.weight") {
            x = ck("up\(ub).up", upsample(x, "decoder.up_blocks.\(ub).upsamplers.0"))
        }
    }
    x = ck("gn_out", silu(gnorm(x, "decoder.conv_norm_out")))
    x = ck("conv_out", conv(x, "decoder.conv_out", 1))
    return x.transposed(0, 3, 1, 2)                        // NCHW [B,3,8h,8w]
}
