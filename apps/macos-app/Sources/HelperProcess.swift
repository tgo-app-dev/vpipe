import Foundation

// A running helper binary, with its output captured line by line.
//
// Stop is SIGINT, not terminate(): both vpipe and vpipe-web-ui install a
// SIGINT handler that drains the pipeline and closes the LMDB
// environment. SIGKILL past a grace period is a backstop for a helper
// that is wedged, and it is deliberately generous -- a video VAE decode
// can sit in one Metal dispatch for a while without being stuck.
@MainActor
final class HelperProcess: ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var lines: [String] = []
    @Published private(set) var exitStatus: Int32?

    private var process: Process?
    private var pipe: Pipe?
    private var stdinPipe: Pipe?
    private var pending = ""

    // A pipeline can be interactive: a stage asks a question through the
    // session's UI delegate, which on the CLI reads a line from stdin.
    // Without a pipe here the child inherits the GUI app's stdin, which
    // is not a terminal, so the read returns EOF immediately and the
    // stage sees an empty answer rather than waiting for one.
    //
    // This gives line-at-a-time input, which covers the prompts vpipe
    // actually issues. It is NOT a terminal: nothing that needs raw key
    // events, cursor addressing or a TTY-only progress display will
    // behave the way it does in a shell. "Run in Terminal" exists for
    // those.
    var acceptsInput: Bool { stdinPipe != nil && isRunning }

    // Bounded so a chatty run cannot grow the log view without limit --
    // a progress bar redrawing for an hour is a lot of lines.
    private let maxLines = 5000

    var pid: Int32? { process?.processIdentifier }

    func start(executable: URL,
               arguments: [String],
               workDir: URL,
               environment: [String: String] = [:]) throws {
        guard !isRunning else { return }

        lines.removeAll()
        pending = ""
        exitStatus = nil

        let p = Process()
        p.executableURL = executable
        p.arguments = arguments
        p.currentDirectoryURL = workDir

        var env = ProcessInfo.processInfo.environment
        // The helpers colour their output when stdout is a TTY. It is a
        // pipe here, so they will not -- but a stale TERM from a parent
        // shell can still reach code that checks it, and escape codes in
        // a SwiftUI Text render as mojibake rather than colour.
        env["NO_COLOR"] = "1"
        env["TERM"] = "dumb"
        for (k, v) in environment { env[k] = v }
        p.environment = env

        let out = Pipe()
        p.standardOutput = out
        p.standardError = out
        pipe = out

        let inp = Pipe()
        p.standardInput = inp
        stdinPipe = inp

        out.fileHandleForReading.readabilityHandler = { [weak self] fh in
            let data = fh.availableData
            guard !data.isEmpty else { return }
            let chunk = String(decoding: data, as: UTF8.self)
            Task { @MainActor in self?.absorb(chunk) }
        }

        p.terminationHandler = { [weak self] proc in
            Task { @MainActor in
                self?.finish(status: proc.terminationStatus)
            }
        }

        try p.run()
        process = p
        isRunning = true
    }

    // Split on newlines, carrying a partial last line across reads --
    // a pipe boundary lands mid-line often enough that not doing this
    // visibly tears the output.
    private func absorb(_ chunk: String) {
        pending += chunk
        while let nl = pending.firstIndex(of: "\n") {
            let line = String(pending[pending.startIndex..<nl])
            pending = String(pending[pending.index(after: nl)...])
            append(line)
        }
        // A progress bar redraws with \r and no newline. Show the latest
        // state rather than nothing at all.
        if pending.contains("\r"), let last = pending.split(separator: "\r").last {
            append(String(last))
            pending = ""
        }
    }

    private func append(_ line: String) {
        lines.append(line)
        if lines.count > maxLines {
            lines.removeFirst(lines.count - maxLines)
        }
    }

    // Send one line to the child's stdin, echoed into the log so the
    // transcript reads the way the same session would in a terminal --
    // a prompt with no visible answer is confusing to come back to.
    func send(_ line: String) {
        guard let inp = stdinPipe, isRunning else { return }
        append("> " + line)
        let data = Data((line + "\n").utf8)
        // write(contentsOf:) throws on a closed pipe, which happens
        // whenever the child exits between the guard and here.
        try? inp.fileHandleForWriting.write(contentsOf: data)
    }

    // Close stdin without stopping the process: the signal a program
    // reading "until EOF" is waiting for.
    func closeInput() {
        try? stdinPipe?.fileHandleForWriting.close()
        stdinPipe = nil
    }

    private func finish(status: Int32) {
        if !pending.isEmpty { append(pending); pending = "" }
        pipe?.fileHandleForReading.readabilityHandler = nil
        pipe = nil
        try? stdinPipe?.fileHandleForWriting.close()
        stdinPipe = nil
        process = nil
        isRunning = false
        exitStatus = status
    }

    func stop(graceSeconds: Double = 30) {
        guard let p = process, p.isRunning else { return }
        let target = p.processIdentifier
        kill(target, SIGINT)
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: UInt64(graceSeconds * 1e9))
            // Re-check identity, not just liveness: by now the original
            // may have exited and the pid been reused.
            if let cur = self.process, cur.isRunning,
               cur.processIdentifier == target {
                kill(target, SIGKILL)
            }
        }
    }

    var logText: String { lines.joined(separator: "\n") }
}
