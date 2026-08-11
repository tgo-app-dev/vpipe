import Foundation
import AppKit

struct PipelineFile: Identifiable, Hashable {
    let url: URL
    // Stage types in this file that have a web-UI panel. Non-empty means
    // the run expects someone at the browser -- a mask to draw, a region
    // to pick -- and launching it here alone would leave it waiting.
    var composerStages: [String] = []
    // Under <work>/sandbox rather than <work> itself. Worth showing:
    // the web UI confines stage file I/O to sandbox/, so a pipeline
    // whose outputs are meant to be reachable from the browser lives
    // there, and one meant for the CLI lives in the work folder.
    let inSandbox: Bool
    var id: String { url.path }
    var name: String { url.deletingPathExtension().lastPathComponent }
    var needsComposer: Bool { !composerStages.isEmpty }
}

// Runs a .vpipeline through Contents/Helpers/vpipe.
//
// The CLI is used rather than the web UI for the same reason the
// MiniMax-H3 guide tells a reader to: a preparation run is long,
// unattended and disk-bound, with nothing to click once it starts, and
// Ctrl-C -- here, the Stop button -- resumes on the next run because the
// fetch stage skips files already on disk at the right size.
@MainActor
final class PipelineRunner: ObservableObject {
    @Published private(set) var running: PipelineFile?
    // Shutting down but not gone yet, so the UI can say so and offer
    // the escalation rather than leaving Stop looking inert.
    @Published private(set) var stopping = false
    let proc = HelperProcess()

    private var stopTask: Task<Void, Never>?

    // Matches the web server's. A stage that checks its stop flag
    // rarely is exactly the case where waiting longer does not help --
    // the run is not finishing, it is not looking.
    static let stopGrace: Double = 15

    var isRunning: Bool { proc.isRunning }

    // Stage types with a web-UI panel, asked of the binary rather than
    // listed here: `vpipe --list-views` reads the same registry the web
    // UI mounts from, so a view added later is picked up without this
    // app knowing about it.
    private lazy var composerStageTypes: Set<String> = {
        let p = Process()
        p.executableURL = BundlePaths.cli
        p.arguments = ["--list-views"]
        let out = Pipe(); p.standardOutput = out; p.standardError = Pipe()
        guard (try? p.run()) != nil else { return [] }
        let d = out.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        struct V: Decodable { let stage_type: String }
        let vs = (try? JSONDecoder().decode([V].self, from: d)) ?? []
        return Set(vs.map(\.stage_type).filter { !$0.isEmpty })
    }()

    // Which composer-backed stage types a spec actually contains.
    //
    // Parsed as plain JSON rather than handed to vpipe: this runs for
    // every file in the list on every refresh, and launching a process
    // per pipeline to answer a question about its text would be absurd.
    private func composerStages(in url: URL) -> [String] {
        guard !composerStageTypes.isEmpty,
              let data = try? Data(contentsOf: url),
              let root = try? JSONSerialization.jsonObject(with: data)
                  as? [String: Any],
              let stages = root["stages"] as? [[String: Any]]
        else { return [] }
        var hits: [String] = []
        for st in stages {
            if let t = st["type"] as? String,
               composerStageTypes.contains(t), !hits.contains(t) {
                hits.append(t)
            }
        }
        return hits
    }

    // Everything under the work directory, plus its sandbox/.
    //
    // Nothing ships inside the app: a pipeline that cannot be edited is
    // not much use (the prepare-* ones need an hf_token filled in), and
    // a file inside a signed bundle cannot be edited without breaking
    // the signature. Both folders are scanned because the two kinds
    // live in different places -- CLI pipelines in the work folder,
    // web-UI ones under sandbox/ where the browser can reach them.
    func discover(workDir: URL) -> [PipelineFile] {
        var out: [PipelineFile] = []
        let fm = FileManager.default

        if let items = try? fm.contentsOfDirectory(
            at: workDir, includingPropertiesForKeys: nil) {
            out += items
                .filter { $0.pathExtension == "vpipeline" }
                .map { PipelineFile(url: $0, composerStages:
                                    composerStages(in: $0),
                                    inSandbox: false) }
        }
        if let items = try? fm.contentsOfDirectory(
            at: workDir.appendingPathComponent("sandbox"),
            includingPropertiesForKeys: nil) {
            out += items
                .filter { $0.pathExtension == "vpipeline" }
                .map { PipelineFile(url: $0, composerStages:
                                    composerStages(in: $0),
                                    inSandbox: true) }
        }
        return out.sorted { $0.name < $1.name }
    }

    func run(_ file: PipelineFile, model: AppModel) throws {
        guard !proc.isRunning else { return }
        try model.ensureWorkDir()

        var args = ["--launch", file.url.path]
        let cfg = model.sessionConfigJSON()
        if cfg != "{}" { args += ["--config", cfg] }

        try proc.start(executable: BundlePaths.cli,
                       arguments: args,
                       workDir: model.workDirURL,
                       environment: model.helperEnvironment)
        running = file
    }

    func stop(graceSeconds: Double = PipelineRunner.stopGrace) {
        guard proc.isRunning else { stopping = false; return }
        stopping = true
        // Arms SIGINT now and SIGKILL after the grace period.
        proc.stop(graceSeconds: graceSeconds)

        stopTask?.cancel()
        stopTask = Task { @MainActor in
            let deadline = Date().addingTimeInterval(graceSeconds + 5)
            while Date() < deadline {
                if !self.proc.isRunning { break }
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
            self.stopping = false
            self.stopTask = nil
        }
    }

    // SIGKILL now, for the user who does not want to wait it out.
    func forceStop() { proc.kill() }

    // MARK: - editing

    static let textEdit = URL(fileURLWithPath:
        "/System/Applications/TextEdit.app")

    func reveal(_ file: PipelineFile) {
        NSWorkspace.shared.activateFileViewerSelecting([file.url])
    }

    // Opened explicitly in TextEdit rather than through the default
    // handler: this app REGISTERS itself as the owner of .vpipeline (see
    // Info.plist), so "open with the default application" would hand the
    // file straight back to itself.
    //
    // Always in place. Nothing ships inside the bundle any more, so
    // every pipeline the list shows is a normal, writable file in the
    // user's own work folder.
    func edit(_ file: PipelineFile) {
        NSWorkspace.shared.open(
            [file.url], withApplicationAt: Self.textEdit,
            configuration: NSWorkspace.OpenConfiguration())
    }
}