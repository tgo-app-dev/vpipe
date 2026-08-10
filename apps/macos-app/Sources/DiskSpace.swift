import Foundation

// Free space on the volume holding the work directory.
//
// Shown when choosing that directory, because the choice is hard to
// undo cheaply: the models this app runs are enormous (MiniMax-H3
// downloads ~115 GB and peaks near ~155 GB while it quantizes), and
// moving tens of gigabytes after the fact is its own afternoon.
//
// This used to back a preflight check that compared a specific model's
// size against the volume before starting a download. That went with
// the model manager; the number by itself is still the thing worth
// seeing at the moment the directory is picked.
enum DiskSpace {

    // .volumeAvailableCapacityForImportantUsage, not
    // volumeAvailableCapacity: on APFS the plain figure ignores space
    // that is purgeable (local snapshots, caches) and so UNDER-reports
    // what a large write can actually have. The "important usage"
    // figure is what the system will free up for exactly this kind of
    // write, and it is what Finder shows -- so it is also the number
    // the user can check against.
    static func availableBytes(at path: String) -> Int64? {
        let url = URL(fileURLWithPath: path)
        // Walk up to the nearest existing ancestor: the work directory
        // may not have been created yet, and a missing path has no
        // volume to ask about.
        var probe = url
        while !FileManager.default.fileExists(atPath: probe.path) {
            let parent = probe.deletingLastPathComponent()
            if parent.path == probe.path { return nil }
            probe = parent
        }
        let values = try? probe.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }

    static func format(_ bytes: Int64) -> String {
        let f = ByteCountFormatter()
        f.countStyle = .file
        f.allowedUnits = [.useGB, .useTB]
        return f.string(fromByteCount: bytes)
    }
}
