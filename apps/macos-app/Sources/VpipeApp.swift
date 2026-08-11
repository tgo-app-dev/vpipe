import SwiftUI
import AppKit

// Order is the sidebar order, and Web UI leads: it is the surface most
// sessions start from, and several pipelines cannot finish without it.
enum Pane: String, CaseIterable, Identifiable {
    case webui     = "Web UI"
    case pipelines = "Pipelines"
    case perms     = "Permissions"
    case settings  = "Settings"
    var id: String { rawValue }

    var icon: String {
        switch self {
        case .webui:     return "globe"
        case .pipelines: return "play.rectangle"
        case .perms:     return "lock.shield"
        case .settings:  return "gearshape"
        }
    }
}

@main
struct VpipeApp: App {
    @StateObject private var model   = AppModel.shared
    @StateObject private var server  = WebUIServer()
    @StateObject private var runner  = PipelineRunner()
    @StateObject private var perms   = Permissions()
    @StateObject private var updater = Updater()
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        Window("Vpipe Manager", id: "main") {
            RootView(model: model, server: server, runner: runner,
                     perms: perms)
                .frame(minWidth: 820, minHeight: 560)
                .onAppear { delegate.server = server; delegate.runner = runner }
        }
        .commands {
            CommandGroup(replacing: .newItem) {}
            if updater.canCheck {
                CommandGroup(after: .appInfo) {
                    Button("Check for Updates…") {
                        updater.checkForUpdates()
                    }
                }
            }
        }

        // Feature 2: the menu-bar item.
        //
        // Earns its place because the work here outlives a window: a
        // preparation run is hours long and the server is meant to stay
        // up while a phone talks to it. This is where the user checks on
        // both without keeping a window in the way.
        MenuBarExtra("Vpipe Manager", systemImage: menuIcon) {
            if let c = server.connection {
                // String(c.port), not "\(c.port)": interpolating an Int
                // into a Text goes through LocalizedStringKey, which
                // formats it as a QUANTITY and renders 9876 as "9,876".
                // A port is an identifier.
                Text("Serving on Port " + String(c.port))
                Text("Access Key: \(c.key)")
                Button("Open in Browser") { server.openInBrowser() }
                Button("Copy Access Key") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(c.key, forType: .string)
                }
                Divider()
                Button("Stop Server") { server.stop() }
            } else if server.stopping {
                Text("Stopping…")
                Button("Force Quit Server") { server.forceStop() }
            } else if server.starting {
                Text("Starting…")
            } else {
                Text("Server Stopped")
                Button("Start Server") { server.start(model: model) }
            }

            Divider()

            if runner.proc.isRunning, let r = runner.running {
                Text("Running \(r.name)")
                Button("Stop Pipeline") { runner.stop() }
            } else {
                Text("No Pipeline Running").foregroundStyle(.secondary)
            }

            Divider()
            Button("Open Vpipe Manager") {
                NSApp.activate(ignoringOtherApps: true)
                NSApp.windows.first?.makeKeyAndOrderFront(nil)
            }
            Button("Quit Vpipe Manager") { NSApp.terminate(nil) }
                .keyboardShortcut("q")
        }
    }

    // Filled while something is live, so the state is readable from the
    // menu bar without opening the menu.
    private var menuIcon: String {
        (server.isRunning || runner.proc.isRunning)
            ? "cube.fill" : "cube"
    }
}

// Closing the window must not quit: a generation or a preparation run
// would go with it. The menu-bar item is how the app is brought back,
// and Quit is how it is ended -- at which point the helpers are stopped
// deliberately rather than orphaned.
final class AppDelegate: NSObject, NSApplicationDelegate {
    var server: WebUIServer?
    var runner: PipelineRunner?

    func applicationShouldTerminateAfterLastWindowClosed(
        _ sender: NSApplication) -> Bool { false }

    func applicationShouldTerminate(
        _ sender: NSApplication) -> NSApplication.TerminateReply {
        let serverUp = server?.isRunning ?? false
        let runUp    = runner?.proc.isRunning ?? false
        guard serverUp || runUp else { return .terminateNow }

        let a = NSAlert()
        a.messageText = runUp
            ? "A pipeline is still running."
            : "The web UI server is still running."
        a.informativeText = runUp
            ? "Quitting stops it. A model preparation run resumes from "
            + "where it left off the next time you start it; a generation "
            + "does not."
            : "Quitting stops the server, and any device connected to it "
            + "will lose access."
        a.addButton(withTitle: "Quit")
        a.addButton(withTitle: "Cancel")
        guard a.runModal() == .alertFirstButtonReturn else {
            return .terminateCancel
        }

        // SIGINT and give the helpers a moment to close LMDB cleanly --
        // terminating out from under an open write transaction is how a
        // registry ends up needing recovery. Hence .terminateLater
        // rather than .terminateNow: the app must outlive the helpers
        // long enough to see them go, or it exits first and leaves an
        // orphaned web server still holding its port.
        //
        // .terminateLater is a PROMISE to call reply(toApplication-
        // ShouldTerminate:) once that is done. Without the reply AppKit
        // waits forever: the app sits in a pending-terminate state where
        // the window still works but Cmd-Q only beeps, because a
        // terminate is already in flight. (Quitting from the menu bar
        // appeared to work only because by then the helpers had stopped,
        // so the guard above returned .terminateNow and never reached
        // this path.)
        runner?.stop(graceSeconds: kQuitGrace)
        server?.stop(graceSeconds: kQuitGrace)
        waitForHelpersThenReply()
        return .terminateLater
    }

    // Long enough for a SIGINT to be handled and LMDB closed, short
    // enough that quitting never feels stuck.
    private let kQuitGrace: Double = 4

    private func waitForHelpersThenReply() {
        Task { @MainActor in
            let deadline = Date().addingTimeInterval(kQuitGrace + 2)
            while Date() < deadline {
                let busy = (runner?.proc.isRunning ?? false)
                        || (server?.isRunning ?? false)
                if !busy { break }
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            // Past the deadline a helper is wedged. Reply anyway rather
            // than refusing to quit -- HelperProcess.stop's own SIGKILL
            // timer has fired by now, and a user who asked twice should
            // not be held hostage by a stuck child.
            NSApp.reply(toApplicationShouldTerminate: true)
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        // Last resort, including paths that never went through
        // applicationShouldTerminate (a logout, say). Usually a no-op,
        // because stop() returns immediately for a process that has
        // already exited. When one has NOT, grace 0 means it is killed
        // rather than left behind: we are exiting either way, and an
        // orphaned vpipe-web-ui still holding its port is worse than an
        // ungraceful stop.
        runner?.stop(graceSeconds: 0)
        server?.stop(graceSeconds: 0)
    }
}

struct RootView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var server: WebUIServer
    @ObservedObject var runner: PipelineRunner
    @ObservedObject var perms: Permissions
    @State private var pane: Pane = .webui

    var body: some View {
        NavigationSplitView {
            List(Pane.allCases, selection: $pane) { p in
                Label(p.rawValue, systemImage: p.icon).tag(p)
            }
            .navigationSplitViewColumnWidth(min: 150, ideal: 170, max: 220)
        } detail: {
            Group {
                switch pane {
                case .pipelines:
                    PipelinePane(model: model, runner: runner,
                                 proc: runner.proc, server: server)
                case .webui:
                    WebUIPane(model: model, server: server, proc: server.proc)
                case .perms:
                    PermissionsPane(perms: perms)
                case .settings:
                    SettingsPane(model: model)
                }
            }
            .navigationTitle(pane.rawValue)
        }
        .overlay(alignment: .top) { missingHelperBanner }
    }

    // A bundle missing its helpers cannot do anything, and every button
    // would otherwise fail one at a time with its own opaque error.
    @ViewBuilder
    private var missingHelperBanner: some View {
        let missing = BundlePaths.missingHelpers
        if !missing.isEmpty {
            Text("This build is incomplete — missing: "
                 + missing.map { ($0 as NSString).lastPathComponent }
                          .joined(separator: ", "))
                .font(.callout)
                .padding(8)
                .frame(maxWidth: .infinity)
                .background(.red.opacity(0.85))
                .foregroundStyle(.white)
        }
    }
}
