import Foundation
import AppKit

struct ConnectionInfo: Decodable {
    let url: String
    let scheme: String
    let host: String
    let port: Int
    let key: String
    let qr_link: String?
    let pid: Int32
}

// Runs vpipe-web-ui and surfaces what a user needs to connect to it.
//
// The connection details come from --emit-connection-json, not from
// parsing stdout. Two of the three are not on stdout to parse: with
// port 0 the bound port is only settled after the listen, and the QR
// link is deliberately never printed (it carries the link secret, which
// is the thing the QR exists to keep out of scrollback).
@MainActor
final class WebUIServer: ObservableObject {
    @Published private(set) var connection: ConnectionInfo?
    @Published private(set) var starting = false
    // Shutting down but not gone yet. Tracked because stop() used to
    // clear the connection and return, so the button flipped straight
    // back to "Start Server" while the process was still alive -- and
    // pressing it did nothing, because start() guards on isRunning.
    @Published private(set) var stopping = false
    @Published private(set) var lastError: String?

    let proc = HelperProcess()

    private var connFile: URL?
    private var pollTask: Task<Void, Never>?
    private var stopTask: Task<Void, Never>?

    // How long a stop waits before SIGKILL.
    //
    // 15s, shorter than the 30 a pipeline's Stop button allows. A
    // pipeline may legitimately sit in one long Metal dispatch. A
    // server that will not shut down is usually wedged BECAUSE a stage
    // is not checking its stop flag often enough, and the server cannot
    // exit until that pipeline drains -- so waiting longer does not
    // make it likelier to finish, it just makes the app look broken.
    static let stopGrace: Double = 15

    var isRunning: Bool { proc.isRunning }

    func start(model: AppModel) {
        guard !proc.isRunning, !starting else { return }
        starting = true
        lastError = nil
        connection = nil

        do {
            try model.ensureWorkDir()
        } catch {
            starting = false
            lastError = "Could not create \(model.workDir): "
                + error.localizedDescription
            return
        }

        // The connection file carries the access key, so it goes in this
        // user's private temp directory, not the work directory the
        // sandbox exposes to stages.
        let f = FileManager.default.temporaryDirectory
            .appendingPathComponent("vpipe-webui-\(UUID().uuidString).json")
        connFile = f
        try? FileManager.default.removeItem(at: f)

        var args = ["--port", String(model.port),
                    "--show-qr",
                    "--emit-connection-json", f.path]
        // Empty means "let vpipe-web-ui choose", which is its own
        // default (this Mac's LAN address) -- so send nothing.
        if !model.bindAddress.isEmpty {
            args += ["--bind", model.bindAddress]
        }
        if model.useTLS { args.append("--tls") }

        // Mirrors vpipe-web-ui's own precedence: --white-list-path and
        // --os-sandbox are both ignored once --expose-native-file-system
        // is given, so send only what will actually take effect rather
        // than flags the server will silently drop.
        if model.exposeNativeFS {
            args.append("--expose-native-file-system")
        } else {
            for p in model.whitelistPaths where !p.isEmpty {
                args += ["--white-list-path", p]
            }
            if model.osSandbox { args.append("--os-sandbox") }
        }
        let cfg = model.sessionConfigJSON()
        if cfg != "{}" { args += ["--config", cfg] }

        do {
            try proc.start(executable: BundlePaths.webUI,
                           arguments: args,
                           workDir: model.workDirURL,
                           environment: model.helperEnvironment)
        } catch {
            starting = false
            lastError = "Could not launch vpipe-web-ui: "
                + error.localizedDescription
            return
        }

        pollTask = Task { @MainActor in await self.awaitConnection(f) }
    }

    // Poll for the file rather than waiting on a fixed delay: startup is
    // fast, but a cold model manager or a busy disk can stretch it, and
    // a fixed delay would either lie or waste time. Give up eventually
    // so a helper that died on a bad --config does not spin forever.
    private func awaitConnection(_ f: URL) async {
        let deadline = Date().addingTimeInterval(30)
        while Date() < deadline {
            if !proc.isRunning && proc.exitStatus != nil { break }
            if let data = try? Data(contentsOf: f),
               let info = try? JSONDecoder().decode(ConnectionInfo.self,
                                                    from: data) {
                connection = info
                starting = false
                return
            }
            try? await Task.sleep(nanoseconds: 150_000_000)
        }
        starting = false
        if connection == nil {
            lastError = proc.exitStatus.map {
                "vpipe-web-ui exited with status \($0) before it began "
                + "listening. See the log below."
            } ?? "vpipe-web-ui did not report a listening address within "
                + "30 seconds."
        }
    }

    func stop(graceSeconds: Double = WebUIServer.stopGrace) {
        pollTask?.cancel()
        pollTask = nil
        starting = false
        // The connection is dead the moment we ask it to stop, whatever
        // the process does next.
        connection = nil

        guard proc.isRunning else { finishStop(); return }

        stopping = true
        // Arms SIGINT now and SIGKILL after the grace period.
        proc.stop(graceSeconds: graceSeconds)

        stopTask?.cancel()
        stopTask = Task { @MainActor in
            // Only observes -- the escalation is already armed above.
            // The extra margin covers the gap between SIGKILL landing
            // and the termination handler running.
            let deadline = Date().addingTimeInterval(graceSeconds + 5)
            while Date() < deadline {
                if !self.proc.isRunning { break }
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
            self.finishStop()
        }
    }

    // SIGKILL right now, for the user who does not want to wait out the
    // grace period.
    func forceStop() {
        proc.kill()
    }

    private func finishStop() {
        stopping = false
        stopTask = nil
        if let f = connFile { try? FileManager.default.removeItem(at: f) }
        connFile = nil
    }

    func openInBrowser() {
        guard let c = connection else { return }
        // Use the QR link when there is one: it authenticates and then
        // redirects, so the browser lands logged in with the key gone
        // from the address bar. Falls back to the plain URL, which for
        // a LAN bind would ask for the key.
        let target = c.qr_link ?? c.url
        if let u = URL(string: target) { NSWorkspace.shared.open(u) }
    }
}
