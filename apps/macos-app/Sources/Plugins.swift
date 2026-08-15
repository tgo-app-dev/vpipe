import Foundation

// A plugin dylib sitting in <work>/plugins.
//
// Identified by its FILE NAME rather than its path: the folder is
// derived from the work directory, so a stored absolute path would
// point at the old place the moment someone moves their workspace,
// and the selection would silently stop applying.
struct PluginFile: Identifiable, Hashable {
    let url: URL
    var id: String { url.path }
    // The full file name, extension included. Two plugins differing only
    // by suffix are two different files, so the extension is part of the
    // identity rather than noise to trim.
    var name: String { url.lastPathComponent }
}

// Lists what is in <work>/plugins. Nothing here loads anything.
//
// That separation is deliberate and matches the runtime's own rule (see
// common/plugins-root.h): dropping a file into a scanned folder must
// not be enough to execute its code. This lists candidates; loading
// happens only because someone ticked a box and then ran a pipeline.
enum PluginCatalog {
    // ACCEPT BOTH .so AND .dylib on macOS.
    //
    // A plugin is a CMake MODULE library, and MODULE targets get `.so`
    // here -- only SHARED targets get `.dylib`. So the platform whose
    // native suffix is .dylib is precisely the one where plugins are
    // .so, and filtering to .dylib lists nothing at all. dlopen does not
    // care about the suffix, so neither does this. Same rule as the web
    // UI's discovery in apps/web-ui/plugin-api.cc.
    static let extensions: Set<String> = ["so", "dylib"]

    static func discover(in dir: URL) -> [PluginFile] {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.isDirectoryKey])
        else { return [] }
        return items
            .filter { extensions.contains($0.pathExtension) }
            .filter { u in
                // Symlinks are followed rather than skipped -- a plugin
                // kept in a build tree and linked in here is a normal way
                // to work on one -- but a .so DIRECTORY (a bundle laid
                // out that way) is not something to hand to dlopen.
                let vals = try? u.resourceValues(forKeys: [.isDirectoryKey])
                return vals?.isDirectory != true
            }
            .map { PluginFile(url: $0) }
            .sorted { $0.name.localizedStandardCompare($1.name)
                      == .orderedAscending }
    }
}
