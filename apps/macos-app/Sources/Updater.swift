import SwiftUI

#if VPIPE_SPARKLE
import Sparkle
#endif

// Feature 6: auto-update, via Sparkle when it was built in.
//
// The whole feature is behind a compile-time flag because Sparkle is an
// optional 15 MB binary dependency (see fetch-sparkle.sh). Both sides of
// the #if present the SAME small surface to the rest of the app -- a
// `canCheck` flag and a `checkForUpdates()` -- so no caller needs its
// own conditional, and a build without Sparkle simply hides the menu
// item instead of growing a second code path.
@MainActor
final class Updater: ObservableObject {
#if VPIPE_SPARKLE
    private let controller: SPUStandardUpdaterController

    init() {
        // startingUpdater: true begins the scheduled background check.
        // Sparkle asks the user for permission on first run before it
        // sends anything, so this does not phone home unprompted.
        controller = SPUStandardUpdaterController(
            startingUpdater: true, updaterDelegate: nil,
            userDriverDelegate: nil)
    }

    var canCheck: Bool { true }
    func checkForUpdates() { controller.checkForUpdates(nil) }

    var automaticallyChecks: Bool {
        get { controller.updater.automaticallyChecksForUpdates }
        set { controller.updater.automaticallyChecksForUpdates = newValue }
    }
#else
    init() {}
    var canCheck: Bool { false }
    func checkForUpdates() {}
    var automaticallyChecks: Bool {
        get { false }
        set { _ = newValue }
    }
#endif
}
