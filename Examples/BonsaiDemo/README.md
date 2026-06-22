# BonsaiDemo — minimal on-device sample app

A minimal SwiftUI app that runs the full Bonsai pipeline on-device: a prompt
field (with an **Examples** menu that prefills it), a one-line progress readout,
and the rendered image (512², 4 steps). The whole generation is one call
(`BonsaiPipeline.generate`); everything else is UI.

The weights are **bundled into the app** — stage the model into `bonsai-model/`
here before generating the project and they ride inside the `.app`. No Finder
copy, no in-app download.

## 1. Stage the weights (on a Mac)

Put all four inputs under `bonsai-model/` in this folder. Three come from the
public PrismML model; the fourth is the VAE artifact from this repo's release
(both covered in the top-level README ["Get the weights"](../../README.md#get-the-weights)):

```bash
cd Examples/BonsaiDemo

# If you already downloaded per the top-level README, clone it in (instant on APFS):
cp -Rc ~/models/bonsai-image bonsai-model

# …or download straight into bonsai-model/:
#   huggingface-cli download prism-ml/bonsai-image-ternary-4B-mlx-2bit --local-dir bonsai-model
#   gh release download v0.1.0 -R duration-ai/bonsai-swift -p vae.safetensors -D bonsai-model
```

`bonsai-model/` must contain (the app's `BonsaiPipeline(modelDir:)` resolves these):

```
bonsai-model/
  transformer-packed-mflux/diffusion_pytorch_model.safetensors   # DiT
  text_encoder-mlx-4bit/model.safetensors                        # Qwen3 encoder
  tokenizer/                                                     # tokenizer
  vae.safetensors                                                # VAE artifact
```

`bonsai-model/` is gitignored — it is never committed.

## 2. Build & run

**Quick way (CLI — no Xcode wizard).** A committed `project.yml` generates a
buildable project with everything wired (the local `BonsaiEngine` dep, iOS 17,
the memory entitlement, and `bonsai-model/` as a bundled resource):

```bash
brew install xcodegen          # one-time
xcodegen generate              # -> BonsaiDemo.xcodeproj (bundles bonsai-model/)
open BonsaiDemo.xcodeproj      # set your Signing team, pick a device, Run
```

> **Signing (first time).** In **BonsaiDemo ▸ Signing & Capabilities**: tick
> *Automatically manage signing*, pick your **Team**, and change the **Bundle
> Identifier** (the `com.yourname.*` placeholder in `project.yml`) to something
> your team can register. Signing is per-developer — the committed value is only
> a placeholder.

> The model rides inside the `.app`, so the first **signed** build is slow
> (~3.7 GB to sign). Compile-check without a device or signing (fast):
>
> ```bash
> xcodebuild -project BonsaiDemo.xcodeproj -scheme BonsaiDemo \
>   -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build
> ```

<details><summary>Or wire it up by hand in Xcode</summary>

1. **File ▸ New ▸ Project ▸ iOS ▸ App**, SwiftUI, name `BonsaiDemo`; delete the
   generated `ContentView.swift` + `…App.swift`.
2. **File ▸ Add Package Dependencies… ▸ Add Local…** → this checkout's
   `bonsai-swift` folder; add the **`BonsaiEngine`** product to the target.
3. Drag `BonsaiDemoApp.swift` + `ContentView.swift` into the target (✓ *Copy items*),
   and drag `bonsai-model` in as a **folder reference** (blue folder) so it's
   copied into the bundle.
4. Target settings: **iOS 17.0** deployment; add the
   **`com.apple.developer.kernel.increased-memory-limit`** entitlement.

</details>

> **Device only.** Needs a real iPhone with **≥ 6 GB RAM, A14 / iPhone 12 Pro or
> newer, iOS 17+**. It won't run in the Simulator. On first deploy, trust your dev
> cert on the phone (Settings ▸ General ▸ VPN & Device Management).

## 3. Generate

Type a prompt — or pick one from the **Examples** menu (top-right) to prefill the
field — and tap **Generate**. It runs at 512², 4 steps (the trained default); the
first image is ~140 s on an A14 (iPhone 12 Pro). The status line tracks
encode → denoise → decode.

If it shows "Weights not found …", `bonsai-model/` was missing an input when you
ran `xcodegen generate` — fix it and regenerate.

The sample is written to be Swift-6-concurrency-clean; if Xcode's Swift 6 mode
still flags something, set the target's *Swift Language Version* to 5.
