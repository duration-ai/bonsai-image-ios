import Foundation

// macOS parity-test runner for the BonsaiEngine library. Dispatch on the first arg:
//   transformer (default) | vae | preproc | e2e | e2egen | pipeline
// (BONSAI_FP32=1 selects the fp32 golden for the vae mode.)
// `pipeline` is the full end-to-end run via BonsaiPipeline(modelDir:) against the
// real downloaded model — the same path a host app takes — and writes a PNG.
switch CommandLine.arguments.dropFirst().first ?? "" {
case "vae": try vaeParity()
case "qwen": try qwenParity()
case "preproc": try preprocParity()
case "e2e": try e2eParity()
case "e2egen": try e2eGenParity()
case "pipeline": try pipelineGen()
default: try transformerParity()
}
