# bonsai-swift

**Run [Bonsai Image](https://github.com/PrismML-Eng/Bonsai-Image-Demo) (FLUX.2
[klein] 4B, ternary-quantized) entirely on an iPhone** — text encoder included,
no server, airplane-mode capable — via a from-scratch [mlx-swift](https://github.com/ml-explore/mlx-swift)
port.

A 4-billion-parameter text encoder (Qwen3-4B) and a 4-billion-parameter ternary
diffusion transformer run **back-to-back through one ~4.6 GB memory budget** on
an **iPhone 12 Pro (A14, 6 GB, 2020)**: ~12 s to encode the prompt, ~130 s to
denoise 4 steps + VAE-decode at 512×512 — **~140 s prompt-to-pixels, fully
offline.**

> Research spike, not a product. It demonstrates a portable method for getting
> someone else's frontier model onto a device — the model science is entirely
> Prism ML's / Black Forest Labs'; this repo is the on-device porting work
> (parity ladders, iOS memory forensics, and routing around an MLX Metal
> conv-kernel GPU fault on A14).

<p align="center">
  <img src="docs/assets/grid-composite-5up.png" alt="Five 512x512 images generated on an iPhone 12 Pro" width="100%">
  <br><em>512×512, 4 steps, generated on-device on an iPhone 12 Pro.</em>
</p>

## Quickstart

New here? Pick your entry point:

- **Run it on your iPhone** — [get the weights](#get-the-weights), then
  [build the sample app](#run-it-on-a-phone) (one `xcodegen` command). Needs a
  ≥ 6 GB iPhone (A14 / iPhone 12 Pro or newer) and your Apple signing team.
- **Verify the port on a Mac** (no device) — [Verify the port](#verify-the-port-mac-parity-harness)
  runs the parity ladder and the full generate path against the real weights.
- **Build on the engine** — [Using the engine](#using-the-engine):
  `BonsaiPipeline(modelDir:)` is the one call from prompt to pixels.

## What's in here

| Path | What it is |
|---|---|
| `Sources/BonsaiEngine/` | the importable library — Qwen3 encoder, ternary DiT, VAE, scheduler, and the `BonsaiPipeline` one-call API |
| `Sources/BonsaiParity/` | a macOS executable that parity-checks each component against Python goldens, plus a full end-to-end `pipeline` run |
| `Examples/BonsaiDemo/` | a minimal SwiftUI sample app — `xcodegen generate`, then build to a device |
| `fixtures/` | golden tensors for the parity harness (gitignored; regenerate from the Bonsai-Image-Demo repo) |
| `docs/assets/` | the on-device sample renders shown above |

## How it works

The whole pipeline runs on the phone GPU, in **sequential residency** — each
model is loaded, used, and **evicted** before the next loads, so peak memory is
the largest single phase, never the sum:

```
prompt ─► Qwen3-4B encode ─► evict ─► ternary DiT (4-step denoise) ─► evict ─► VAE decode ─► image
         (~2.3 GB)                    (~1.4 GB)                                 (untiled)
```

Everything is **parity-validated** against the Python reference (mflux-prism)
before it's trusted: the quantized matmul is bit-exact; transformer / VAE /
preprocessing / end-to-end all match to cosine > 0.999; tokenizer ids are exact.

## Requirements

- **Device:** an iPhone with **≥ 6 GB RAM on iOS 17+** (A14 / iPhone 12 Pro is
  the proven floor — 4 GB devices can't fit the weights + working set). The
  parity harness runs on any Apple-Silicon Mac.
- **Xcode** with the iOS 17 SDK (mlx-swift 0.31.4 requires it).
- The model weights (below — not bundled here).

## Get the weights

The pipeline needs four inputs. **Three come straight from Prism ML's public
model (Apache-2.0), no conversion** — download it with the Hugging Face CLI:

```bash
huggingface-cli download prism-ml/bonsai-image-ternary-4B-mlx-2bit \
  --local-dir ~/models/bonsai-image
```

That gives you, under the model directory:

| Engine input | File in the download |
|---|---|
| DiT transformer (ternary 2-bit) | `transformer-packed-mflux/diffusion_pytorch_model.safetensors` |
| Qwen3-4B encoder (4-bit, group-64) | `text_encoder-mlx-4bit/model.safetensors` |
| tokenizer | `tokenizer/` |

The **fourth input is the VAE** — the FLUX.2 small decoder
([`black-forest-labs/FLUX.2-small-decoder`](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder),
Apache-2.0), remapped to this engine's MLX key layout. Grab the prebuilt
`vae.safetensors` (~56 MB) from this repo's releases into the same directory:

```bash
gh release download v0.1.0 -R duration-ai/bonsai-swift \
  -p vae.safetensors -D ~/models/bonsai-image
# -> ~/models/bonsai-image/vae.safetensors
```

`BonsaiPipeline(modelDir:)` resolves all four from that one folder.

## Verify the port (Mac parity harness)

SwiftPM can't compile Metal shaders, so build the harness with `xcodebuild`:

```bash
xcodebuild build -scheme BonsaiParity -configuration Release \
  -destination 'platform=macOS' -derivedDataPath .xcbuild -skipPackagePluginValidation

export BONSAI_MODEL=~/models/bonsai-image
B=./.xcbuild/Build/Products/Release/BonsaiParity
$B            # 25-block transformer      $B vae   (BONSAI_FP32=1 for the fp32 golden)
$B preproc    # RoPE / grid / time-embed  $B qwen  # tokenizer ids exact + encoder embeds
$B e2egen     # full generate path
$B pipeline   # FULL end-to-end via BonsaiPipeline -> /tmp/bonsai-out.ppm (needs vae.safetensors in $BONSAI_MODEL)
```

Golden fixtures live in `fixtures/` (gitignored — regenerate them with the
`make_*_fixture.py` scripts in the Bonsai-Image-Demo repo; `BONSAI_FIXTURES`
overrides the path).

## Using the engine

`BonsaiEngine` is the importable library. The one-call path — point it at the
downloaded model directory and generate:

```swift
import BonsaiEngine

let pipeline = BonsaiPipeline(modelDir: "…/bonsai-image")   // resolves all 4 inputs
let (rgb, w, h) = try pipeline.generate(
    prompt: "a meticulously pruned bonsai tree, soft studio lighting, photorealistic",
    height: 512, width: 512, steps: 4, seed: 0,
    progress: { phase, frac in print(phase, frac) })        // row-major HWC uint8 RGB
```

It runs the full flow in **sequential residency** — encode → evict → denoise →
evict → decode — so call it off the main thread. `pipeline.weightsPresent()`
pre-flights the four files; `BonsaiPipeline.memorySnapshotMB()` reads MLX +
jetsam memory for a live readout.

<details><summary>Or drive the stages yourself (the lower-level API)</summary>

```swift
// 1. Encode the prompt, then free the encoder before the DiT loads.
let enc = try Qwen3Encoder(weightsPath: qwenSafetensors)
let ids = QwenTokenizer.promptIds(try QwenTokenizer.load(dir: tokenizerDir), prompt: prompt)
let (promptEmbeds, textIds) = enc.encode(tokenIds: ids)
enc.evict()

// 2. Denoise + decode (evict the 1.4 GB transformer before the VAE decode).
let gen = try BonsaiGenerator(transformerWeightsPath: ditSafetensors, vaeWeightsPath: vaeSafetensors)
let image = gen.generateImageTensor(
    promptEmbeds: promptEmbeds, textIds: textIds,
    height: 512, width: 512, steps: 4, seed: 0,
    vaeFp32: false, evict: true)
let (rgb, w, h) = gen.toRGB8(image)   // row-major HWC uint8 RGB
```
</details>

## Run it on a phone

[`Examples/BonsaiDemo/`](Examples/BonsaiDemo) is a minimal SwiftUI sample — a
prompt field with an **Examples** menu of ready-made prompts, and the rendered
image. It builds from the CLI via [XcodeGen](https://github.com/yonaskolb/XcodeGen),
with the weights **bundled into the app** (no Finder copy, no in-app download):

```bash
brew install xcodegen
cd Examples/BonsaiDemo
cp -Rc ~/models/bonsai-image bonsai-model   # weights from "Get the weights" — bundled in
xcodegen generate                            # -> BonsaiDemo.xcodeproj
open BonsaiDemo.xcodeproj                     # set your Signing team, pick a device, ⌘R
```

The model rides inside the `.app`, so the first **signed** build is slow (~3.7 GB
to sign). Device requirements, the memory entitlement, and a fast no-sign compile
check are in the [sample's README](Examples/BonsaiDemo/README.md).

Optional opt-in instrumentation (off by default) writes a per-phase,
fsync'd memory log that survives an iOS OOM kill:

```swift
BonsaiDebug.memLog = true   // -> <Documents>/bonsai-memlog.txt
```

## Scope & constraints

- **Spike, not a product.** One fixed config (512², 4 steps); the sample bundles
  the weights into the app (a ~3.7 GB resource) rather than streaming them; no
  batching, no scheduler options.
- **Untiled VAE** — fine at 512² on a 6 GB device, but tiling would be needed
  for ≥ 768²–1024² (and to reach 4 GB devices). Image size is gated by the
  transformer's per-step cost long before the VAE, anyway.
- **An MLX Metal conv kernel page-faults on A14** at ≥ 256²-spatial shapes
  (a GPU "BIF0" fault that surfaces as a `std::terminate`/SIGABRT, present in
  mlx-swift 0.25.x–0.31.x); this port routes those convs through an im2col +
  matmul path to avoid the faulting kernel entirely (`VAE.swift`).

## Credits & license

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The model and its
science are **Prism ML's**, built on **FLUX.2 [klein]** (Black Forest Labs) and
**Qwen3** (Alibaba Cloud) — all Apache-2.0. This repository is only the
Swift / mlx-swift inference port and is not affiliated with or endorsed by them.
*Created using Bonsai Image by Prism ML.*
