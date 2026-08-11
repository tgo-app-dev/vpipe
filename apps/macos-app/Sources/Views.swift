import SwiftUI
import AppKit

// The console output of a helper process: selectable, monospaced, and
// non-wrapping.
//
// An NSTextView rather than a SwiftUI Text, for one reason.
// vpipe-web-ui draws its QR code with half-block characters
// (U+2588/2580/2584), packing TWO module rows into every line of text.
// That only survives if consecutive lines touch, and SwiftUI lays text
// out at the font's default line height -- ascent + descent + leading
// -- where the leading is empty space. lineSpacing(0) does not remove
// it, because leading is not line spacing. Only a paragraph style that
// pins the line height to the em box does.
//
// Not a matter of taste: MEASURED, a CIDetector pass over the SwiftUI
// rendering found NO code at all, and over this one decodes the link.
//
// Only the tail is rendered; see maxRendered.
private struct ConsoleText: NSViewRepresentable {
    let lines: [String]
    // Usually a prompt: shown so a question is visible while unanswered.
    let partial: String
    static let maxRendered = 4000

    // The line height at which half-block characters tile with no seam.
    //
    // SwiftUI's Text (even at lineSpacing 0) lays lines out at the
    // font's default line height, which is ascent + descent + LEADING.
    // The leading is empty space, so every second module row of the QR
    // gets a pale stripe through it and the symbol stops decoding --
    // MEASURED: a CIDetector pass over the rendered view found no code
    // at all. A terminal has no such gap because its cell height is
    // exactly the em box.
    //
    // Pinning minimum == maximum line height to ascent + |descent|
    // reproduces the terminal's cell and makes the blocks contiguous.
    private static func paragraphStyle(_ font: NSFont) -> NSParagraphStyle {
        let p = NSMutableParagraphStyle()
        let h = (font.ascender - font.descender).rounded(.up)
        p.minimumLineHeight = h
        p.maximumLineHeight = h
        p.lineSpacing = 0
        p.lineBreakMode = .byClipping
        return p
    }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSTextView.scrollableTextView()
        guard let tv = scroll.documentView as? NSTextView else { return scroll }
        tv.isEditable = false
        tv.isSelectable = true
        tv.isRichText = false
        tv.drawsBackground = true
        tv.backgroundColor = .textBackgroundColor
        // No implicit insets: they would offset the symbol but, more
        // importantly, a horizontal one shifts every line and makes the
        // quiet zone uneven.
        tv.textContainerInset = NSSize(width: 6, height: 6)
        tv.textContainer?.lineFragmentPadding = 0
        // Never wrap. A wrapped QR row is a broken QR row, and wrapped
        // log lines are harder to read than scrolled ones.
        tv.textContainer?.widthTracksTextView = false
        tv.textContainer?.containerSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude)
        tv.isHorizontallyResizable = true
        tv.isVerticallyResizable = true
        tv.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude,
                            height: CGFloat.greatestFiniteMagnitude)
        scroll.hasHorizontalScroller = true
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        scroll.borderType = .noBorder
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        guard let tv = scroll.documentView as? NSTextView else { return }
        let shown = lines.count > Self.maxRendered
            ? Array(lines.suffix(Self.maxRendered)) : lines
        var body = shown.joined(separator: "\n")
        if !partial.isEmpty {
            body += (body.isEmpty ? "" : "\n") + partial
        }
        if lines.count > Self.maxRendered {
            body = "… \(lines.count - Self.maxRendered) earlier lines not "
                 + "shown\n" + body
        }

        let font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        let attr = NSAttributedString(string: body, attributes: [
            .font: font,
            .foregroundColor: NSColor.textColor,
            .paragraphStyle: Self.paragraphStyle(font),
        ])

        // Only rewrite when the text actually changed: replacing the
        // storage drops the user's selection, and a running pipeline
        // updates this several times a second.
        if tv.string != body {
            let atBottom = isScrolledToBottom(scroll)
            tv.textStorage?.setAttributedString(attr)
            tv.sizeToFit()
            if atBottom { tv.scrollToEndOfDocument(nil) }
        }
    }

    // Follow the tail only when the user is already at the tail --
    // yanking the view back down while they are reading further up is
    // the standard way a log viewer becomes unusable.
    private func isScrolledToBottom(_ scroll: NSScrollView) -> Bool {
        let doc = scroll.documentVisibleRect
        guard let h = scroll.documentView?.bounds.height else { return true }
        return doc.maxY >= h - 4
    }
}

// A collapsible console/terminal block.
//
// Collapsed to its title bar by default: for most runs the output is
// reassurance rather than information, and a log occupying half the
// window pushes the controls that matter off the top.
struct ConsoleBlock: View {
    let title: String
    @ObservedObject var proc: HelperProcess
    @Binding var expanded: Bool
    // Shown only when the process is running and can take a line.
    var acceptsInput: Bool = false

    @State private var inputText = ""

    var body: some View {
        VStack(spacing: 0) {
            header
            if expanded {
                Divider()
                // ConsoleText is its own NSScrollView (it has to be, to
                // scroll horizontally without wrapping), so it is placed
                // directly rather than inside a SwiftUI ScrollView.
                ConsoleText(lines: proc.lines, partial: proc.partialLine)
                    .frame(minHeight: 180, idealHeight: 260)
                if acceptsInput && proc.isRunning {
                    Divider()
                    inputRow
                }
            }
        }
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(Color(nsColor: .separatorColor), lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private var header: some View {
        HStack(spacing: 6) {
            Image(systemName: expanded ? "chevron.down" : "chevron.right")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Image(systemName: "terminal")
                .foregroundStyle(.secondary)
            Text(title).font(.callout.weight(.medium))
            if proc.isRunning {
                ProgressView().controlSize(.small).scaleEffect(0.7)
            }
            Spacer()
            Text(proc.lines.isEmpty ? "no output"
                                    : "\(proc.lines.count) lines")
                .font(.caption).foregroundStyle(.secondary)
            if !proc.lines.isEmpty {
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(proc.logText,
                                                   forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(.borderless)
                .help("Copy the console output")
            }
            Button(expanded ? "Collapse" : "Expand") {
                withAnimation(.easeInOut(duration: 0.15)) { expanded.toggle() }
            }
            .buttonStyle(.borderless)
            .font(.caption)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(Color(nsColor: .windowBackgroundColor))
        .contentShape(Rectangle())
        .onTapGesture {
            withAnimation(.easeInOut(duration: 0.15)) { expanded.toggle() }
        }
    }

    private var inputRow: some View {
        HStack(spacing: 6) {
            Image(systemName: "chevron.right")
                .font(.caption).foregroundStyle(.secondary)
            TextField("Type a reply and press Return", text: $inputText)
                .textFieldStyle(.plain)
                .font(.system(size: 11, design: .monospaced))
                .onSubmit(submit)
            Button("Send", action: submit)
                .disabled(inputText.isEmpty)
                .font(.caption)
            Button("EOF") { proc.closeInput() }
                .font(.caption)
                .help("Close standard input (Ctrl-D)")
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private func submit() {
        guard !inputText.isEmpty else { return }
        proc.send(inputText)
        inputText = ""
    }
}

// MARK: - Work directory + session config

struct SettingsPane: View {
    @ObservedObject var model: AppModel
    @State private var freeSpace: String = ""
    @State private var copied: String?
    @State private var showLicenses = false
    @State private var interfaces: [NetInterface] = []

    var body: some View {
        Form {
            Section("Work Directory") {
                // LabeledContent puts "Path" in the left column, on the
                // same baseline grid as the rows below it -- the shape
                // System Settings uses for a field with a label.
                LabeledContent("Path") {
                    HStack {
                        TextField("", text: $model.workDir)
                            .textFieldStyle(.roundedBorder)
                        Button("Choose…") { chooseWorkDir() }
                    }
                }
                Text("vpipe uses this directory as its workspace: "
                     + "**models/** for everything downloaded or quantized, "
                     + "**data.mdb** / **lock.mdb** for the registry and "
                     + "logs, and **sandbox/** for files the web UI lets "
                     + "stages write. Pick a volume with room — prepared "
                     + "models run to tens of gigabytes.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if !freeSpace.isEmpty {
                    Text(freeSpace).font(.caption).foregroundStyle(.secondary)
                }
            }

            Section("Session") {
                Picker("Log Level", selection: $model.logLevel) {
                    ForEach(LogLevel.allCases) { Text($0.display).tag($0) }
                }
                zeroableField("Worker Threads", $model.numWorkers,
                              hint: "0 = one per core")
                zeroableField("Memory Cap (MB)", $model.memoryCapMB,
                              hint: "0 = uncapped. Over the cap the "
                                  + "least-recently-used model weights are "
                                  + "parked, not refused.")
                zeroableField("LMDB Map Size (MB)", $model.dbMapSizeMB,
                              hint: "0 = default")
                zeroableField("Default Edge Capacity", $model.defaultEdgeCapacity,
                              hint: "0 = default (4)")
                // The locales the session actually has translations for
                // (common/i18n.cc). A free-text field could name one
                // that does not exist, which silently falls back rather
                // than reporting anything.
                Picker("Language", selection: $model.language) {
                    ForEach(AppModel.languages, id: \.tag) { l in
                        Text(l.name).tag(l.tag)
                    }
                }
            }

            Section("FFmpeg") {
                LabeledContent("Library Path") {
                    HStack {
                        TextField("", text: $model.ffmpegDir,
                                  prompt: Text("Default"))
                            .textFieldStyle(.roundedBorder)
                        Button("Choose…") { chooseFFmpegDir() }
                        Button("Use Default") { model.ffmpegDir = "" }
                            .disabled(model.ffmpegDir.isEmpty)
                    }
                }
                let probe = AppModel.inspectFFmpegDir(model.ffmpegDir)
                Label(probe.detail, systemImage: probe.ok
                      ? "checkmark.circle" : "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(probe.ok ? Color.secondary : Color.red)
                    .fixedSize(horizontal: false, vertical: true)
                Text("The copy inside the app is a minimal LGPL build, so "
                     + "it has no libx264/x265 and few extra decoders. Point "
                     + "this at a fuller install — Homebrew's is normally "
                     + "**/opt/homebrew/lib** — to use that instead. Takes "
                     + "effect the next time a pipeline or the server is "
                     + "started.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Section("Web UI") {
                Picker("Bind To", selection: $model.bindAddress) {
                    Text("Automatic (this Mac's LAN address)").tag("")
                    Text("This Mac only (127.0.0.1)").tag("127.0.0.1")
                    if !interfaces.isEmpty {
                        Divider()
                        ForEach(interfaces.filter { !$0.isLoopback }) { i in
                            Text(i.label).tag(i.address)
                        }
                    }
                }
                if model.bindIsRemote {
                    Label {
                        Text("Other devices on this network will be able to "
                             + "reach the web UI. It can start and stop "
                             + "pipelines, browse files in the sandbox and "
                             + "drive models on this Mac — an access key is "
                             + "the only thing between the network and that "
                             + "control. Prefer **This Mac only** unless you "
                             + "actually need a phone or another computer to "
                             + "connect, and avoid it on networks you do not "
                             + "trust.")
                    } icon: {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                    }
                    .font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
                } else {
                    Text("Only this Mac can reach the server, and no access "
                         + "key is required.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                // grouping(.never): a port is an identifier, not a
                // quantity. The default format renders 9876 as "9,876",
                // which is not something anyone types into a browser.
                TextField("Port", value: $model.port,
                          format: .number.grouping(.never))
                Toggle("HTTPS (Self-Signed)", isOn: $model.useTLS)
                Text("Needed for the low-latency Preview view on other "
                     + "devices: browsers only allow WebCodecs in a secure "
                     + "context. The certificate is self-signed, so the "
                     + "browser shows a one-time warning to accept.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Web UI File Access") {
                Toggle("Allow Access to the Whole Filesystem",
                       isOn: $model.exposeNativeFS)
                Text(model.exposeNativeFS
                     ? "Stages can read and write anywhere you can. "
                     + "Anyone who reaches the web UI has that reach too."
                     : "Stage file I/O is confined to the **sandbox** "
                     + "folder in your work directory. Model files are "
                     + "exempt.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                LabeledContent("Allowed Folders") {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(Array(model.whitelistPaths.enumerated()),
                                id: \.offset) { i, p in
                            HStack {
                                Text(p).font(.caption)
                                    .truncationMode(.middle).lineLimit(1)
                                Spacer()
                                Button {
                                    model.whitelistPaths.remove(at: i)
                                } label: { Image(systemName: "minus.circle") }
                                .buttonStyle(.borderless)
                            }
                        }
                        Button("Add Folder…") { addWhitelistPath() }
                    }
                }
                .disabled(model.exposeNativeFS)

                Text("Real folders outside the sandbox that stages may "
                     + "reach by their real path. Ignored while access to "
                     + "the whole filesystem is allowed.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                Toggle("Add an OS Sandbox (Seatbelt)", isOn: $model.osSandbox)
                    .disabled(model.exposeNativeFS)
                Text("A whole-process write backstop underneath the "
                     + "confinement above. It cannot be nested, so it "
                     + "**disables the run_python chat tool**, which "
                     + "sandboxes each call itself. Enable it only if you "
                     + "do not use that tool.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Section("About") {
                LabeledContent("Version", value: AppModel.versionString)
                Text("The digits after ***** count files that were "
                     + "uncommitted when this build was made.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                LabeledContent("Command-Line Tools") {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("The app drives these two binaries; the same "
                             + "commands work in a terminal.")
                            .font(.caption).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                        HStack {
                            Button("Copy vpipe Path") {
                                copyToPasteboard(BundlePaths.cli.path)
                            }
                            Button("Copy vpipe-web-ui Path") {
                                copyToPasteboard(BundlePaths.webUI.path)
                            }
                        }
                        if let copied {
                            Text("Copied \(copied)")
                                .font(.caption).foregroundStyle(.secondary)
                                .truncationMode(.middle).lineLimit(1)
                        }
                    }
                }
                LabeledContent("Legal") {
                    VStack(alignment: .leading, spacing: 4) {
                        Button("Third-Party Notices…") { showLicenses = true }
                        Text("This app bundles FFmpeg and other open-source "
                             + "software. FFmpeg is used under the LGPL and "
                             + "may be replaced with your own build — see "
                             + "the notices, and the FFmpeg setting above.")
                            .font(.caption).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .sheet(isPresented: $showLicenses) { LicensesView() }
        // Same reason as PermissionsPane: this pane's ideal width is
        // whatever its longest explanatory paragraph needs unwrapped
        // (MEASURED at 744pt), and NavigationSplitView collapses the
        // sidebar when the detail's ideal will not fit beside it.
        //
        // 744 is under the 820 window minimum only until you subtract
        // the sidebar, so this one was latent rather than absent -- it
        // needed a window at its minimum size to show up.
        .frame(minWidth: 340, idealWidth: 560, maxWidth: .infinity,
               maxHeight: .infinity)
        .onAppear { updateFree(); interfaces = NetworkInterfaces.ipv4() }
        .onChange(of: model.workDir) { _ in updateFree() }
    }

    @ViewBuilder
    private func zeroableField(_ label: String, _ value: Binding<Int>,
                               hint: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            TextField(label, value: value, format: .number)
            Text(hint).font(.caption).foregroundStyle(.secondary)
        }
    }

    private func updateFree() {
        if let b = DiskSpace.availableBytes(at: model.workDir) {
            freeSpace = "\(DiskSpace.format(b)) free on this volume."
        } else {
            freeSpace = ""
        }
    }

    private func copyToPasteboard(_ s: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(s, forType: .string)
        copied = s
    }

    private func addWhitelistPath() {
        let p = NSOpenPanel()
        p.canChooseDirectories = true
        p.canChooseFiles = false
        p.allowsMultipleSelection = true
        p.prompt = "Allow This Folder"
        if p.runModal() == .OK {
            for u in p.urls where !model.whitelistPaths.contains(u.path) {
                model.whitelistPaths.append(u.path)
            }
        }
    }

    private func chooseFFmpegDir() {
        let p = NSOpenPanel()
        p.canChooseDirectories = true
        p.canChooseFiles = false
        p.prompt = "Use This Folder"
        p.message = "Choose the lib/ directory containing libavcodec.dylib "
                  + "and friends."
        if !model.ffmpegDir.isEmpty {
            p.directoryURL = URL(fileURLWithPath: model.ffmpegDir)
        }
        if p.runModal() == .OK, let u = p.url { model.ffmpegDir = u.path }
    }

    private func chooseWorkDir() {
        let p = NSOpenPanel()
        p.canChooseDirectories = true
        p.canChooseFiles = false
        p.canCreateDirectories = true
        p.prompt = "Use as Work Directory"
        p.directoryURL = URL(fileURLWithPath: model.workDir)
        if p.runModal() == .OK, let u = p.url { model.workDir = u.path }
    }
}

// MARK: - Pipelines

struct PipelinePane: View {
    @ObservedObject var model: AppModel
    @ObservedObject var runner: PipelineRunner
    @ObservedObject var proc: HelperProcess
    // Only so the composer confirmation can offer to start the server;
    // the pane does nothing else with it.
    @ObservedObject var server: WebUIServer
    @State private var files: [PipelineFile] = []
    @State private var selection: PipelineFile?
    @State private var consoleExpanded = false
    @State private var confirmComposer = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading) {
                HStack {
                    Text("Pipelines").font(.headline)
                    Spacer()
                    Button("Add…") { addPipeline() }
                    Button("Reload") { reload() }
                }
                List(files, selection: $selection) { f in
                    pipelineRow(f).tag(f)
                }
                controlsBar
                terminalHintRow
            }
            .padding()

            ConsoleBlock(title: "Console", proc: proc,
                         expanded: $consoleExpanded, acceptsInput: true)
                .padding(.horizontal)
                .padding(.bottom)
        }
        // Launching one of these from here is not wrong, but it will sit
        // waiting for a person at the browser, and from this pane that
        // looks like a hang. Say so once, and offer the thing that
        // actually unblocks it.
        .confirmationDialog(composerTitle,
                            isPresented: $confirmComposer,
                            titleVisibility: .visible) {
            Button("Start Web UI and Run") {
                if !server.isRunning { server.start(model: model) }
                if let s = selection { try? runner.run(s, model: model) }
            }
            Button("Run Anyway") {
                if let s = selection { try? runner.run(s, model: model) }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(composerMessage)
        }
        // Pin the ideal width, for the reason spelled out on
        // PermissionsPane: NavigationSplitView sizes its columns from
        // the detail's ideal, and a detail that wants more than the
        // window can spare beside the sidebar gets it by COLLAPSING the
        // sidebar. Every pane carries this, not just the ones measured
        // over budget today -- a pane's width depends on runtime state
        // (a connection, a long path, a status line), so "it fits right
        // now" is not a property that stays true.
        .frame(minWidth: 340, idealWidth: 560, maxWidth: .infinity,
               maxHeight: .infinity, alignment: .topLeading)
        .onAppear(perform: reload)
        .onChange(of: model.workDir) { _ in reload() }
    }

    // Hand the run to Terminal.app.
    //
    // Written to a small script rather than passed as a command string:
    // the work directory and the pipeline path are user-chosen and can
    // contain spaces or quotes, and building a shell line out of them by
    // interpolation is how a path with an apostrophe becomes a syntax
    // error -- or worse, two commands.
    private func runInTerminal(_ file: PipelineFile) {
        try? model.ensureWorkDir()
        let cfg = model.sessionConfigJSON()
        var script = "#!/bin/bash\n"
        if !model.ffmpegDir.isEmpty {
            script += "export VPIPE_FFMPEG_DIR="
                    + shellQuote(model.ffmpegDir) + "\n"
        }
        script += "cd " + shellQuote(model.workDir) + " || exit 1\n"
        script += shellQuote(BundlePaths.cli.path)
        script += " --launch " + shellQuote(file.url.path)
        if cfg != "{}" { script += " --config " + shellQuote(cfg) }
        script += "\n"

        let dir = FileManager.default.temporaryDirectory
        let sh = dir.appendingPathComponent("vpipe-run-\(UUID().uuidString).sh")
        do {
            try script.write(to: sh, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o700], ofItemAtPath: sh.path)
        } catch {
            return
        }
        NSWorkspace.shared.open(
            [sh],
            withApplicationAt: URL(fileURLWithPath: "/System/Applications/"
                                   + "Utilities/Terminal.app"),
            configuration: NSWorkspace.OpenConfiguration())
    }

    // POSIX single-quote quoting: wrap in single quotes and replace each
    // embedded quote with '\'' -- the only form with no escape
    // characters to reason about inside.
    private func shellQuote(_ s: String) -> String {
        "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    private var composerTitle: String {
        (selection?.name ?? "This pipeline") + " needs the web UI"
    }

    private var composerMessage: String {
        let which = selection?.composerStages.joined(separator: ", ") ?? ""
        return "It contains " + which + ", which draws its own panel in "
             + "the web UI. Run on its own, the pipeline will start and "
             + "then wait for input that has nowhere to arrive from."
    }

    // Extracted for the same reason as the row: body as one expression
    // exceeded what the type checker will solve.
    @ViewBuilder
    private var controlsBar: some View {
        HStack {
            Button(proc.isRunning ? "Running…" : "Run") {
                guard let s = selection else { return }
                if s.needsComposer { confirmComposer = true }
                else { try? runner.run(s, model: model) }
            }
            .disabled(selection == nil || proc.isRunning)
            .keyboardShortcut(.return)

            Button(runner.stopping ? "Stopping…" : "Stop") { runner.stop() }
                .disabled(!proc.isRunning || runner.stopping)

            if runner.stopping {
                Button("Force Quit") { runner.forceStop() }
                Text("Force-quit automatically after "
                     + "\(Int(PipelineRunner.stopGrace))s.")
                    .font(.caption).foregroundStyle(Color.secondary)
            }

            Spacer()
            if let s = proc.exitStatus, !proc.isRunning {
                Text(s == 0 ? "Finished." : "Exited with status \(s).")
                    .font(.caption)
                    .foregroundStyle(s == 0 ? Color.secondary : Color.red)
            }
            Button("Open Work Folder") {
                NSWorkspace.shared.open(model.workDirURL)
            }
        }
    }

    // For anything that genuinely needs a terminal. The console below
    // takes a line at a time, which covers the prompts vpipe issues, but
    // it is a pipe and not a TTY.
    @ViewBuilder
    private var terminalHintRow: some View {
        HStack {
            Button("Run in Terminal…") {
                if let s = selection { runInTerminal(s) }
            }
            .disabled(selection == nil)
            Text("Runs the same command in Terminal, for pipelines that "
                 + "need a real terminal.")
                .font(.caption).foregroundStyle(Color.secondary)
            Spacer()
        }
    }

    // Extracted from the List closure: inline, the row grew past what
    // SwiftUI's type checker will solve ("unable to type-check this
    // expression in reasonable time").
    @ViewBuilder
    private func pipelineRow(_ f: PipelineFile) -> some View {
        HStack(spacing: 6) {
            Image(systemName: f.inSandbox ? "shippingbox" : "doc.text")
                .foregroundStyle(.secondary)
            Text(f.name)
            if f.needsComposer { composerBadge(f) }
            if f.inSandbox { sandboxBadge() }
            if runner.running?.id == f.id, proc.isRunning {
                ProgressView().controlSize(.small).scaleEffect(0.6)
            }
            Spacer()
            // .borderless so a click lands on the button rather than
            // being swallowed as a row selection.
            Button { revealFile(f) } label: { Image(systemName: "folder") }
                .buttonStyle(.borderless)
                .help("Show in Finder")
            Button { editFile(f) } label: {
                Image(systemName: "square.and.pencil")
            }
            .buttonStyle(.borderless)
            .help("Open in TextEdit")
        }
        .contextMenu {
            Button("Edit in TextEdit") { editFile(f) }
            Button("Show in Finder") { revealFile(f) }
        }
    }

    @ViewBuilder
    private func composerBadge(_ f: PipelineFile) -> some View {
        Label("composer", systemImage: "hand.tap")
            .font(.caption2)
            .foregroundStyle(Color.orange)
            .padding(.horizontal, 4)
            .padding(.vertical, 1)
            .background(Color.orange.opacity(0.15))
            .clipShape(Capsule())
            .help("Contains " + f.composerStages.joined(separator: ", ")
                  + " — needs the web UI to finish.")
    }

    @ViewBuilder
    private func sandboxBadge() -> some View {
        Text("sandbox")
            .font(.caption2)
            .foregroundStyle(Color.secondary)
            .padding(.horizontal, 4)
            .padding(.vertical, 1)
            .background(Color(nsColor: .separatorColor).opacity(0.5))
            .clipShape(Capsule())
    }

    private func reload() {
        files = runner.discover(workDir: model.workDirURL)
        if let sel = selection,
           !files.contains(where: { $0.id == sel.id }) { selection = nil }
    }

    private func revealFile(_ f: PipelineFile) {
        runner.reveal(f)
    }

    // Editing a BUNDLED pipeline yields a copy in the work folder, so
    // the list and the selection have to follow it there -- otherwise
    // the user edits one file and then runs a different one.
    private func editFile(_ f: PipelineFile) {
        runner.edit(f)
    }

    // Copy into the work directory rather than running in place: a
    // pipeline's relative paths resolve against the working directory,
    // so one living elsewhere would behave differently from the same
    // file listed here.
    private func addPipeline() {
        let p = NSOpenPanel()
        p.allowedContentTypes = [.json]
        p.allowsOtherFileTypes = true
        p.canChooseFiles = true
        if p.runModal() == .OK, let u = p.url {
            let dst = model.workDirURL.appendingPathComponent(u.lastPathComponent)
            try? model.ensureWorkDir()
            try? FileManager.default.removeItem(at: dst)
            try? FileManager.default.copyItem(at: u, to: dst)
            reload()
        }
    }
}

// MARK: - Web UI + QR

struct WebUIPane: View {
    @ObservedObject var model: AppModel
    @ObservedObject var server: WebUIServer
    @ObservedObject var proc: HelperProcess
    @State private var consoleExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Button(server.stopping ? "Stopping…"
                           : (server.isRunning ? "Stop Server" : "Start Server")) {
                        server.isRunning ? server.stop() : server.start(model: model)
                    }
                    .disabled(server.starting || server.stopping)
                    if server.starting || server.stopping {
                        ProgressView().controlSize(.small)
                    }
                    // A wedged shutdown is the case this exists for: a
                    // stage that checks its stop flag rarely keeps the
                    // pipeline draining, and the server cannot exit
                    // until it does. Rather than make the user wait out
                    // the grace period wondering whether it is stuck,
                    // offer the escalation directly.
                    if server.stopping {
                        Button("Force Quit") { server.forceStop() }
                        Text("Waiting for the server to shut down. It "
                             + "will be force-quit automatically after "
                             + "\(Int(WebUIServer.stopGrace)) seconds.")
                            .font(.caption).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer()
                    Button("Open Work Folder") {
                        NSWorkspace.shared.open(model.workDirURL)
                    }
                    // The sandbox is created by vpipe-web-ui at startup,
                    // so before the first run there is nothing to open.
                    // Create it rather than failing silently -- the user
                    // asking for it is usually about to put a pipeline
                    // or an input file in it.
                    Button("Open Sandbox Folder") {
                        try? FileManager.default.createDirectory(
                            at: model.sandboxURL,
                            withIntermediateDirectories: true)
                        NSWorkspace.shared.open(model.sandboxURL)
                    }
                    Button("Open in Browser") { server.openInBrowser() }
                        .disabled(server.connection == nil)
                }

                // Shown next to the Start button, not only in Settings:
                // this is where the decision is acted on, and a warning
                // filed under a settings pane the user visited once is
                // not a warning at the moment it matters.
                if model.bindIsRemote {
                    HStack(alignment: .top, spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Reachable from other devices on this "
                                 + "network")
                                .font(.callout.weight(.medium))
                            Text("Anyone who reaches this server and has the "
                                 + "access key can start and stop pipelines, "
                                 + "browse the sandbox and drive models on "
                                 + "this Mac. Only enable it on a network you "
                                 + "trust; change it under Settings ▸ Web UI.")
                                .font(.caption).foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.orange.opacity(0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                }

                if let e = server.lastError {
                    Text(e).foregroundStyle(.red).font(.callout)
                }

                if let c = server.connection {
                    HStack(alignment: .top, spacing: 20) {
                        VStack(alignment: .leading, spacing: 8) {
                            LabeledContent("Address") {
                                Text(c.url).textSelection(.enabled)
                                    .font(.system(.body, design: .monospaced))
                            }
                            LabeledContent("Access Key") {
                                Text(c.key).textSelection(.enabled)
                                    .font(.system(.title3, design: .monospaced))
                                    .bold()
                            }
                            Text("This Mac connects without a key. Other "
                                 + "devices must supply it.")
                                .font(.caption).foregroundStyle(.secondary)
                            if c.qr_link != nil {
                                Text("Scanning the code opens the UI already "
                                     + "signed in and removes the key from "
                                     + "the address bar. The code is a "
                                     + "secret in a link — it is only as "
                                     + "private as this screen.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                        // Let the text column absorb the slack instead
                        // of every label demanding its unwrapped width.
                        // Without this the pane's ideal width was
                        // MEASURED at 1036pt once a connection existed
                        // -- the text column at its widest PLUS a
                        // 190pt QR -- which does not fit beside the
                        // sidebar and made NavigationSplitView collapse
                        // it. Same failure as the Permissions pane had,
                        // reached through runtime state rather than
                        // static content.
                        .frame(maxWidth: .infinity, alignment: .leading)

                        if let link = c.qr_link, let img = QR.image(for: link) {
                            VStack(spacing: 4) {
                                Image(nsImage: img)
                                    .interpolation(.none)
                                    .resizable()
                                    .frame(width: 190, height: 190)
                                Text("Scan with a phone")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                } else if !server.isRunning {
                    Text("Start the server to get a link, an access key and "
                         + "a QR code for connecting a phone.")
                        .foregroundStyle(.secondary)
                }
            }
            .padding()

            // The server prints its own QR to stdout as half-block
            // characters. ConsoleBlock renders those correctly; the
            // native code above is still the one to scan.
            ConsoleBlock(title: "Console", proc: proc,
                         expanded: $consoleExpanded)
                .padding(.horizontal)
                .padding(.bottom)
        }
        // Pin the ideal width, for the reason spelled out on
        // PermissionsPane: NavigationSplitView sizes its columns from
        // the detail's ideal, and a detail that wants more than the
        // window can spare beside the sidebar gets it by COLLAPSING the
        // sidebar. Every pane carries this, not just the ones measured
        // over budget today -- a pane's width depends on runtime state
        // (a connection, a long path, a status line), so "it fits right
        // now" is not a property that stays true.
        .frame(minWidth: 340, idealWidth: 560, maxWidth: .infinity,
               maxHeight: .infinity, alignment: .topLeading)
    }
}

// MARK: - Permissions

struct PermissionsPane: View {
    @ObservedObject var perms: Permissions

    var body: some View {
        VStack(alignment: .leading) {
            Text("These are the permissions the pipeline stages need. "
                 + "Granting them here grants them to vpipe itself — run "
                 + "from a terminal instead, macOS attributes them to the "
                 + "terminal application.")
                .font(.callout).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.bottom, 4)

            // A ScrollView of rows rather than a List.
            //
            // There are four rows, fixed at compile time, none of them
            // selectable or reorderable -- a List buys nothing here. It
            // also costs something: a List row combining a Spacer (which
            // proposes unbounded width) with a Text asking to size itself
            // vertically for whatever width it is given is a layout the
            // AppKit-backed List is known to resolve badly on macOS. The
            // explicit maxWidth below pins the one dimension that was
            // ambiguous, so the height question has a single answer.
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    ForEach(perms.rows) { row in
                        permissionRow(row)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 4)
            }

            HStack {
                Spacer()
                Button("Check Again") { perms.refresh() }
            }
        }
        .padding()
        // Pin the IDEAL width.
        //
        // Without this the pane is rigid: asked for its ideal size it
        // reports the width at which none of its text wraps, which
        // MEASURED at 1053pt. NavigationSplitView sizes its columns from
        // the detail's ideal, the window minimum is 820, and a detail
        // that wants 1053 alongside a 170pt sidebar does not fit -- so
        // the split view resolves it the only way it can, by COLLAPSING
        // the sidebar. That is the "navigation items disappeared" bug:
        // not a hang, not a crash, just a column squeezed to nothing
        // while the divider stayed draggable.
        //
        // A frame here rather than tweaking each Text: the ideal width
        // is a property of the pane as a whole, and any future label
        // long enough to widen it would silently bring the bug back.
        .frame(minWidth: 340, idealWidth: 520, maxWidth: .infinity,
               maxHeight: .infinity, alignment: .topLeading)
    }

    @ViewBuilder
    private func permissionRow(_ row: PermissionRow) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: icon(row.state))
                .foregroundStyle(color(row.state))
                .frame(width: 20)

            VStack(alignment: .leading, spacing: 2) {
                Text(row.title).bold()
                Text(row.detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            VStack(alignment: .trailing, spacing: 4) {
                if let req = row.request {
                    Button("Request") { Task { await req() } }
                }
                if let s = row.settingsURL {
                    Button("Settings…") { Permissions.openSettings(s) }
                        .buttonStyle(.link)
                }
            }
            .fixedSize()
        }
    }

    private func icon(_ s: PermState) -> String {
        switch s {
        case .granted:      return "checkmark.circle.fill"
        case .denied:       return "xmark.circle.fill"
        case .undetermined: return "questionmark.circle"
        case .unknown:      return "circle.dashed"
        }
    }

    private func color(_ s: PermState) -> Color {
        switch s {
        case .granted:      return .green
        case .denied:       return .red
        case .undetermined: return .orange
        case .unknown:      return .secondary
        }
    }
}
