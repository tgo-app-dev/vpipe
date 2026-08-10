import SwiftUI
import AppKit

// The third-party notices, shown in the app.
//
// Not strictly required -- LGPL-2.1 section 6 asks for "prominent
// notice with each copy of the work" and a copy of the licence, which
// files inside the bundle satisfy. But a notice nobody can find is a
// thin reading of "prominent", and the whole thing costs one button, so
// the texts ship BOTH as files in Contents/Resources/Licenses and here.
//
// Read off disk rather than compiled in: those files are the artifact
// that actually accompanies the binary, so displaying anything else
// would let the two drift apart and show a notice that is not the one
// being distributed.
struct LicenseDoc: Identifiable, Hashable {
    let url: URL
    var id: String { url.path }
    var name: String { url.lastPathComponent }

    // NOTICE first -- it is the summary that explains the rest.
    var sortKey: String {
        name.hasPrefix("NOTICE") ? "0" + name : "1" + name
    }
}

enum Licenses {
    static var directory: URL {
        Bundle.main.bundleURL
            .appendingPathComponent("Contents/Resources/Licenses")
    }

    static func documents() -> [LicenseDoc] {
        let items = (try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)) ?? []
        return items
            .filter { ["txt", "md"].contains($0.pathExtension) }
            .map { LicenseDoc(url: $0) }
            .sorted { $0.sortKey < $1.sortKey }
    }
}

struct LicensesView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var docs: [LicenseDoc] = []
    @State private var selection: LicenseDoc?
    @State private var text = ""

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Third-Party Notices").font(.headline)
                Spacer()
                Button("Show in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting(
                        [Licenses.directory])
                }
                Button("Done") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
            .padding()

            Divider()

            if docs.isEmpty {
                // A bundle with no notices is a packaging bug, and one
                // with legal consequences -- say so rather than showing
                // an empty list that reads as "nothing to declare".
                VStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text("This build carries no licence files.")
                    Text("Contents/Resources/Licenses is missing or empty.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                HSplitView {
                    List(docs, selection: $selection) { d in
                        Text(d.name).font(.callout).tag(d)
                    }
                    .frame(minWidth: 200, idealWidth: 220, maxWidth: 280)

                    ScrollView([.vertical, .horizontal]) {
                        Text(text)
                            .font(.system(size: 11, design: .monospaced))
                            .textSelection(.enabled)
                            .padding(10)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(minWidth: 420)
                }
            }
        }
        .frame(minWidth: 720, minHeight: 460)
        .onAppear {
            docs = Licenses.documents()
            selection = docs.first
        }
        .onChange(of: selection) { _ in load() }
        .onChange(of: docs.count) { _ in load() }
    }

    private func load() {
        guard let d = selection else { text = ""; return }
        text = (try? String(contentsOf: d.url, encoding: .utf8))
            ?? "Could not read \(d.name)."
    }
}
