import Foundation
import MLX

// FlowMatchEuler discrete scheduler (sigma-shift path used by Klein, since
// requires_sigma_shift -> set_image_seq_len). Ported from
// flux2 common/schedulers/flow_match_euler_discrete_scheduler.py:
//   sigmas = time_shift_exp(mu, linspace(1, 1/n, n)); timesteps = sigmas*1000;
//   sigmas += [0]. Euler step = latents + (sigma[t+1]-sigma[t])·noise.
// Verified to reproduce the e2e fixture sigmas [1.0,0.955,0.875,0.701,0.0] @ 4 steps, seq 64.

public struct FlowMatchScheduler {
    public let sigmas: [Float]       // length n+1 (terminal 0 appended)
    public let timesteps: [Float]    // length n

    public init(numSteps n: Int, imageSeqLen: Int, numTrain: Float = 1000) {
        let mu = FlowMatchScheduler.empiricalMu(imageSeqLen: imageSeqLen, numSteps: n)
        let expMu = exp(mu)
        var s = (0 ..< n).map { i -> Float in  // linspace(1, 1/n, n)
            let t = 1.0 - (1.0 - 1.0 / Float(n)) * Float(i) / Float(n - 1)
            return expMu / (expMu + (1.0 / t - 1.0))   // time_shift_exponential(mu, 1, t)
        }
        timesteps = s.map { $0 * numTrain }
        s.append(0.0)
        sigmas = s
    }

    // piecewise-linear in num_steps, anchored at the seq-len-dependent slopes a1/a2.
    static func empiricalMu(imageSeqLen: Int, numSteps: Int) -> Float {
        let a1: Float = 8.73809524e-05, b1: Float = 1.89833333
        let a2: Float = 0.00016927, b2: Float = 0.45666666
        let L = Float(imageSeqLen)
        if L > 4300 { return a2 * L + b2 }
        let m200 = a2 * L + b2, m10 = a1 * L + b1
        let a = (m200 - m10) / 190.0
        let b = m200 - 200.0 * a
        return a * Float(numSteps) + b
    }

    // one Euler step on packed latents.
    public func step(_ latents: MLXArray, noise: MLXArray, t: Int) -> MLXArray {
        latents + (sigmas[t + 1] - sigmas[t]) * noise
    }
}
