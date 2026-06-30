import SwiftUI
import UIKit
import BonsaiEngine

// Minimal on-device sample: prompt -> Generate -> image (512², 4 steps).
// Weights live under <Documents>/bonsai-model/ — see README.md.

@MainActor
final class BonsaiModel: ObservableObject {
    // Weights are bundled into the app: drop the model into
    // Examples/BonsaiDemo/bonsai-model/ before generating the project (see README).
    // The `??` is just a non-existent path so the "weights not found" message shows
    // if you build the app without staging them.
    let modelDir: URL = Bundle.main.url(forResource: "bonsai-model", withExtension: nil)
        ?? Bundle.main.bundleURL.appendingPathComponent("bonsai-model")

    @Published var prompt = "a meticulously pruned bonsai tree in a ceramic pot, soft studio lighting, photorealistic"
    @Published var status = ""
    @Published var isRunning = false
    @Published var image: UIImage?

    func generate() {
        guard !isRunning else { return }
        let pipeline = BonsaiPipeline(modelDir: modelDir.path)
        guard pipeline.weightsPresent() else {
            status = "Weights not found under \(modelDir.path) — see README."
            return
        }
        isRunning = true; image = nil; status = "starting…"
        let prompt = self.prompt, dir = modelDir.path

        Task.detached(priority: .userInitiated) {
            do {
                let out = try BonsaiPipeline(modelDir: dir).generate(
                    prompt: prompt, height: 512, width: 512, steps: 4, seed: 0,
                    progress: { phase, frac in
                        Task { @MainActor in self.status = "\(phase.rawValue) \(Int(frac * 100))%" }
                    })
                let rgb = out.rgb, w = out.width, h = out.height   // Sendable
                await MainActor.run {
                    self.image = BonsaiModel.makeImage(rgb: rgb, width: w, height: h)
                    self.status = "done"; self.isRunning = false
                }
            } catch {
                await MainActor.run { self.status = "failed: \(error.localizedDescription)"; self.isRunning = false }
            }
        }
    }

    // row-major HWC uint8 RGB -> UIImage (expand to RGBA).
    nonisolated static func makeImage(rgb: Data, width: Int, height: Int) -> UIImage? {
        let n = width * height
        var rgba = [UInt8](repeating: 255, count: n * 4)
        rgb.withUnsafeBytes { raw in
            let s = raw.bindMemory(to: UInt8.self)
            for i in 0 ..< n { rgba[i*4] = s[i*3]; rgba[i*4+1] = s[i*3+1]; rgba[i*4+2] = s[i*3+2] }
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let cg = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                               bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                               provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { return nil }
        return UIImage(cgImage: cg)
    }
}

struct ContentView: View {
    @StateObject private var model = BonsaiModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                TextField("Prompt", text: $model.prompt, axis: .vertical)
                    .lineLimit(2...4)
                    .textFieldStyle(.roundedBorder)
                    .disabled(model.isRunning)

                Button(action: model.generate) {
                    HStack {
                        if model.isRunning { ProgressView() }
                        Text(model.isRunning ? "Generating…" : "Generate").bold()
                    }
                    .frame(maxWidth: .infinity).padding(.vertical, 4)
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.isRunning || model.prompt.isEmpty)

                if !model.status.isEmpty {
                    Text(model.status).font(.footnote).foregroundStyle(.secondary)
                }
                if let image = model.image {
                    Image(uiImage: image).resizable().scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                Spacer()
            }
            .padding()
            .navigationTitle("Bonsai on-device")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu("Examples") {
                        ForEach(PresetPrompts.all, id: \.name) { preset in
                            Button(preset.name) { model.prompt = preset.text }
                        }
                    }
                    .disabled(model.isRunning)
                }
            }
        }
    }
}

enum PresetPrompts {
    /// Sample prompts that prefill the field (the comparison-grid subjects).
    static let all: [(name: String, text: String)] = [
        ("Bonsai tree", "A bonsai tree in a quiet ceramic studio, soft morning light, shallow depth of field"),
        ("Humpback whale", "A massive humpback whale breaching beside a tiny fishing boat, dramatic ocean spray"),
        ("Jellyfish", "A bioluminescent jellyfish ballet in dark ocean depths, ethereal and otherworldly"),
        ("Mountain cabin", "A cozy mountain cabin in winter storm, smoke from chimney, warm windows, romantic landscape"),
        ("Weathered sailor", "A weathered sailor in oilskin coat, salt spray on his beard, golden hour photography"),
    ]
}

#Preview { ContentView() }
