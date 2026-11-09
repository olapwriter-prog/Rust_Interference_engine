// swift-tools-version:5.9
// Swift package for the Rust Inference Engine. Build the static library first:
//   ../../scripts/build-ios.sh          (CPU)  -> build/RIE.xcframework
//   ../../scripts/build-ios.sh --gpu    (CPU + Metal)
import PackageDescription

let package = Package(
    name: "RIE",
    platforms: [.iOS(.v15), .macOS(.v12)],
    products: [.library(name: "RIE", targets: ["RIE"])],
    targets: [
        .binaryTarget(name: "RieNative", path: "build/RIE.xcframework"),
        .target(name: "CRie", dependencies: ["RieNative"], path: "Sources/CRie"),
        .target(name: "RIE", dependencies: ["CRie"], path: "Sources/RIE",
                linkerSettings: [.linkedFramework("Metal"), .linkedFramework("QuartzCore")]),
    ]
)
