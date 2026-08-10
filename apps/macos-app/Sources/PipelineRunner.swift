import Foundation
import AppKit

struct PipelineFile: Identifiable, Hashable {
    let url: URL
    // Under <work>/sandbox rather than <work> itself. Worth showing:
    // the web UI confines stage file I/O to sandbox/, so a pipeline
    // whose outputs are meant to be reachable from the browser lives
    // there, and one meant for the CLI lives in the work folder.
    let inSandbox: Bool
    var id: String { url.path }
    var name: String { url.deletingPathExtension().lastPathComponent }
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
    let proc = HelperProcess()

    var isRunning: Bool { proc.isRunning }

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
                .map { PipelineFile(url: $0, inSandbox: false) }
        }
        if let items = try? fm.contentsOfDirectory(
            at: workDir.appendingPathComponent("sandbox"),
            includingPropertiesForKeys: nil) {
            out += items
                .filter { $0.pathExtension == "vpipeline" }
                .map { PipelineFile(url: $0, inSandbox: true) }
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

    func stop(graceSeconds: Double = 30) {
        proc.stop(graceSeconds: graceSeconds)
    }

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