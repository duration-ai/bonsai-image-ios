// swift-tools-version: 6.0
import PackageDescription

// Bonsai-Image on-device port (mlx-swift). BonsaiEngine = the reusable generator
// library (iOS + macOS, consumed by the Expo native module). BonsaiParity = the
// macOS executable that parity-tests the engine against Python fixtures.
let package = Package(
    name: "bonsai-swift",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "BonsaiEngine", targets: ["BonsaiEngine"]),
    ],
    dependencies: [
        // 0.31.4: the 0.25.x Metal conv kernel page-faults on A14 at 96ch/256²
        // (BIF0 fault -> check_error throws in the completion handler -> SIGABRT).
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
        // Qwen3 BPE tokenization (tokenizer.json) for the on-device text encoder.
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.3.3")),
    ],
    targets: [
        .target(
            name: "BonsaiEngine",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "BonsaiParity",
            dependencies: ["BonsaiEngine"]
        ),
    ]
)
