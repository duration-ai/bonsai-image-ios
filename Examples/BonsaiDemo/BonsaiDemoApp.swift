import SwiftUI

/// Minimal host app for the BonsaiEngine on-device pipeline.
/// See README.md in this folder for the (one-time) Xcode setup + getting weights.
@main
struct BonsaiDemoApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}
