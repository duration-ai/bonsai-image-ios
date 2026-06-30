import Foundation
import MLX

// Preprocessing builders (validated by preprocParity): 4-axis RoPE (pos_embed),
// the latent grid ids, and time_guidance_embed. Shared by the denoise driver.

// pos_embed: per axis i, omega = theta^(-arange(0,dim,2)/dim); out = pos ⊗ omega; cos/sin; concat axes.
public func posEmbedRope(_ ids: MLXArray) -> (MLXArray, MLXArray) {  // [seq,4] -> [seq,64] cos,sin
    let theta: Float = 2000, dim = 32, half = dim / 2
    var coss = [MLXArray](), sins = [MLXArray]()
    let omega = exp(MLXArray((0 ..< half).map { Float($0 * 2) / Float(dim) }) * (-logf(theta)))  // [16]
    for i in 0 ..< 4 {
        let pos = ids[0..., i].asType(.float32)                      // [seq]
        let out = pos.reshaped([-1, 1]) * omega.reshaped([1, -1])    // [seq,16]
        coss.append(cos(out)); sins.append(sin(out))
    }
    return (concatenated(coss, axis: -1), concatenated(sins, axis: -1))
}

// concat[txt,img] rope for the joint sequence (txt first), matching the transformer forward.
public func buildRope(imgIds: MLXArray, txtIds: MLXArray) -> (MLXArray, MLXArray) {
    let (icos, isin) = posEmbedRope(imgIds)
    let (tcos, tsin) = posEmbedRope(txtIds)
    return (concatenated([tcos, icos], axis: 0), concatenated([tsin, isin], axis: 0))
}

// prepare_grid_ids: per-token (t=0, h, w, layer=0), w fastest (idx = h*W + w). [H*W,4] int32.
public func latentGridIds(_ H: Int, _ Wd: Int) -> MLXArray {
    var rows = [Int32]()
    for h in 0 ..< H { for x in 0 ..< Wd { rows += [0, Int32(h), Int32(x), 0] } }
    return MLXArray(rows, [H * Wd, 4])
}

// time_guidance_embed: sinusoidal(256, flip_sin_to_cos) -> linear_1 -> silu -> linear_2 (no bias).
public func timestepEmbedding(_ t: MLXArray, _ dim: Int) -> MLXArray {  // [B] -> [B,dim]
    let half = dim / 2
    let freqs = exp(MLXArray((0 ..< half).map { Float($0) }) * (-logf(10000.0) / Float(half)))  // [half]
    let args = t.reshaped([-1, 1]) * freqs.reshaped([1, -1])        // [B,half]
    let emb = concatenated([sin(args), cos(args)], axis: -1)        // [B,dim]
    return concatenated([emb[0..., half ..< dim], emb[0..., 0 ..< half]], axis: -1)  // flip_sin_to_cos
}

public func timeGuidanceEmbed(_ timestep: Float, _ w: (String) -> MLXArray) -> MLXArray {  // -> [1,3072]
    let emb = timestepEmbedding(MLXArray([timestep]), 256)
    func dense(_ k: String, _ a: MLXArray) -> MLXArray { matmul(a, w(k).transposed()) }
    let base = "time_guidance_embed.timestep_embedder"
    let h = dense("\(base).linear_1.weight", emb)
    return dense("\(base).linear_2.weight", h * sigmoid(h))
}
