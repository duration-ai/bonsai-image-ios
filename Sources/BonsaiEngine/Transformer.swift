import Foundation
import MLX
import MLXFast

// Reusable 25-block Klein transformer (5 double + 20 single), extracted from the
// validated parity harness so the denoise driver can call it in a loop. Modulation
// is shared across blocks (single weight per kind, no block index) and depends only
// on temb, so it's computed once per forward. Validated by transformerParity().

struct DMod { let wMsa, bMsa, gMsa, wMlp, bMlp, gMlp: MLXArray }  // double-stream modulation
struct SMod { let w, b, gate: MLXArray }                          // single-stream modulation

public struct KleinTransformer {
    let W: [String: MLXArray]
    let B: Int
    let DIM = 3072, H = 24, D = 128, N_DOUBLE = 5, N_SINGLE = 20
    let EPS: Float = 1e-6
    var SCALE: Float { 1.0 / Float(D).squareRoot() }

    public init(W: [String: MLXArray], B: Int) { self.W = W; self.B = B }

    func w(_ k: String) -> MLXArray { guard let v = W[k] else { fatalError("model missing \(k)") }; return v }
    func silu(_ a: MLXArray) -> MLXArray { a * sigmoid(a) }
    func dense(_ key: String, _ a: MLXArray) -> MLXArray { matmul(a, w("\(key).weight").transposed()) }
    func qlin(_ key: String, _ a: MLXArray) -> MLXArray {  // [B,seq,in] -> [B,seq,out]
        let seq = a.shape[1]
        let y = quantizedMatmul(a.reshaped([B * seq, a.shape[2]]).asType(.bfloat16),
            w("\(key).weight"), scales: w("\(key).scales"), biases: w("\(key).biases"),
            transpose: true, groupSize: 128, bits: 2)
        return y.reshaped([B, seq, -1])
    }
    func rms(_ a: MLXArray, _ key: String) -> MLXArray {
        MLXFast.rmsNorm(a, weight: w("\(key).weight").asType(.bfloat16), eps: EPS)
    }
    func applyRope(_ a: MLXArray, _ rcos: MLXArray, _ rsin: MLXArray) -> MLXArray {
        let dt = a.dtype, af = a.asType(.float32)
        let b = af.shape[0], h = af.shape[1], s = af.shape[2], dh = rcos.shape[1]
        let cosB = rcos.reshaped([1, 1, s, dh]).asType(.float32)
        let sinB = rsin.reshaped([1, 1, s, dh]).asType(.float32)
        let pr = split(af.reshaped([b, h, s, dh, 2]), parts: 2, axis: -1)
        let re = pr[0].reshaped([b, h, s, dh]), im = pr[1].reshaped([b, h, s, dh])
        return stacked([re * cosB - im * sinB, im * cosB + re * sinB], axis: -1).reshaped(af.shape).asType(dt)
    }
    func toBHSD(_ a: MLXArray, _ seq: Int) -> MLXArray { a.reshaped([B, seq, H, D]).transposed(0, 2, 1, 3) }
    func sdpa(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ seq: Int) -> MLXArray {
        MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: SCALE, mask: nil)
            .transposed(0, 2, 1, 3).reshaped([B, seq, DIM])
    }
    func ff3(_ a: MLXArray, _ inK: String, _ outK: String) -> MLXArray {
        let p = split(qlin(inK, a), parts: 2, axis: 2)
        return qlin(outK, silu(p[0]) * p[1])
    }
    func modsSingle(_ key: String, _ temb: MLXArray) -> SMod {
        let t = split(matmul(silu(temb), w("\(key).weight").transposed()).reshaped([B, 3, DIM]), parts: 3, axis: 1)
        return SMod(w: 1.0 + t[1].reshaped([DIM]), b: t[0].reshaped([DIM]), gate: t[2].reshaped([DIM]))
    }
    func modsDouble(_ key: String, _ temb: MLXArray) -> DMod {  // split is MSA vs MLP (img/txt is the key)
        let two = split(matmul(silu(temb), w("\(key).weight").transposed()).reshaped([B, 2, 3, DIM]), parts: 2, axis: 1)
        func trip(_ z: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
            let t = split(z.reshaped([B, 3, DIM]), parts: 3, axis: 1)
            return (1.0 + t[1].reshaped([DIM]), t[0].reshaped([DIM]), t[2].reshaped([DIM]))
        }
        let (w0, b0, g0) = trip(two[0]), (w1, b1, g1) = trip(two[1])
        return DMod(wMsa: w0, bMsa: b0, gMsa: g0, wMlp: w1, bMlp: b1, gMlp: g1)
    }

    func doubleBlock(_ img: MLXArray, _ txt: MLXArray, _ i: Int, _ mImg: DMod, _ mTxt: DMod,
                     _ rcos: MLXArray, _ rsin: MLXArray) -> (MLXArray, MLXArray) {
        let a = "transformer_blocks.\(i).attn", bp = "transformer_blocks.\(i)"
        let iSeq = img.shape[1], tSeq = txt.shape[1]
        let nImg = MLXFast.layerNorm(img, weight: mImg.wMsa, bias: mImg.bMsa, eps: EPS)
        let nTxt = MLXFast.layerNorm(txt, weight: mTxt.wMsa, bias: mTxt.bMsa, eps: EPS)
        // rmsnorm per-stream (position-independent); RoPE on the CONCATENATED [txt,img] seq.
        let iq = rms(toBHSD(qlin("\(a).to_q", nImg), iSeq), "\(a).norm_q")
        let ik = rms(toBHSD(qlin("\(a).to_k", nImg), iSeq), "\(a).norm_k")
        let iv = toBHSD(qlin("\(a).to_v", nImg), iSeq)
        let tq = rms(toBHSD(qlin("\(a).add_q_proj", nTxt), tSeq), "\(a).norm_added_q")
        let tk = rms(toBHSD(qlin("\(a).add_k_proj", nTxt), tSeq), "\(a).norm_added_k")
        let tv = toBHSD(qlin("\(a).add_v_proj", nTxt), tSeq)
        let fq = applyRope(concatenated([tq, iq], axis: 2), rcos, rsin)
        let fk = applyRope(concatenated([tk, ik], axis: 2), rcos, rsin)
        let attn = sdpa(fq, fk, concatenated([tv, iv], axis: 2), tSeq + iSeq)
        let ap = split(attn, indices: [tSeq], axis: 1)
        let ctxAttn = qlin("\(a).to_add_out", ap[0])
        let outAttn = qlin("\(a).to_out.0", ap[1])
        var h = img + mImg.gMsa * outAttn
        h = h + mImg.gMlp * ff3(MLXFast.layerNorm(h, weight: mImg.wMlp, bias: mImg.bMlp, eps: EPS),
                                "\(bp).ff.linear_in", "\(bp).ff.linear_out")
        var e = txt + mTxt.gMsa * ctxAttn
        e = e + mTxt.gMlp * ff3(MLXFast.layerNorm(e, weight: mTxt.wMlp, bias: mTxt.bMlp, eps: EPS),
                                "\(bp).ff_context.linear_in", "\(bp).ff_context.linear_out")
        return (e, h)
    }

    func singleBlock(_ hidden: MLXArray, _ i: Int, _ m: SMod, _ rcos: MLXArray, _ rsin: MLXArray) -> MLXArray {
        let a = "single_transformer_blocks.\(i).attn"
        let S = hidden.shape[1]
        let fused = qlin("\(a).to_qkv_mlp_proj", MLXFast.layerNorm(hidden, weight: m.w, bias: m.b, eps: EPS))
        let fs = split(fused, indices: [3 * DIM], axis: 2)
        let q3 = split(fs[0], parts: 3, axis: 2)
        let q = applyRope(rms(toBHSD(q3[0], S), "\(a).norm_q"), rcos, rsin)
        let k = applyRope(rms(toBHSD(q3[1], S), "\(a).norm_k"), rcos, rsin)
        let attnOut = sdpa(q, k, toBHSD(q3[2], S), S)
        let mp = split(fs[1], parts: 2, axis: 2)
        let blockOut = qlin("\(a).to_out", concatenated([attnOut, silu(mp[0]) * mp[1]], axis: -1))
        return hidden + m.gate * blockOut
    }

    // latents [B,imgSeq,128] + textEmb [B,txtSeq,7680] + temb [B,3072] -> noise [B,imgSeq,128]
    public func forward(latents: MLXArray, textEmb: MLXArray, temb: MLXArray,
                        ropeCos: MLXArray, ropeSin: MLXArray) -> MLXArray {
        let mImg = modsDouble("double_stream_modulation_img.linear", temb)
        let mTxt = modsDouble("double_stream_modulation_txt.linear", temb)
        let mS = modsSingle("single_stream_modulation.linear", temb)
        var img = dense("x_embedder", latents)
        var txt = dense("context_embedder", textEmb)
        for i in 0 ..< N_DOUBLE { (txt, img) = doubleBlock(img, txt, i, mImg, mTxt, ropeCos, ropeSin) }
        var hidden = concatenated([txt, img], axis: 1)
        for i in 0 ..< N_SINGLE { hidden = singleBlock(hidden, i, mS, ropeCos, ropeSin) }
        let txtSeq = textEmb.shape[1]
        let imgOut0 = split(hidden, indices: [txtSeq], axis: 1)[1]
        let nmod = split(dense("norm_out.linear", silu(temb)), parts: 2, axis: -1)
        let imgOut = MLXFast.layerNorm(imgOut0, weight: 1.0 + nmod[0].reshaped([DIM]),
                                       bias: nmod[1].reshaped([DIM]), eps: EPS)
        return dense("proj_out", imgOut)
    }
}
