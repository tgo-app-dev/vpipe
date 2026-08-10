import Foundation
import SwiftUI

// Where the bundle keeps the two command-line binaries it drives.
//
// VPIPE_HELPER_DIR overrides the lookup so the GUI can be run straight
// out of a build tree during development, where there is no bundle
// around it.
enum BundlePaths {
    static var helperDir: URL {
        if let override = ProcessInfo.processInfo.environment["VPIPE_HELPER_DIR"],
           !override.isEmpty {
            return URL(fileURLWithPath: override)
        }
        return Bundle.main.bundleURL
            .appendingPathComponent("Contents/Helpers")
    }

    static var cli: URL { helperDir.appendingPathComponent("vpipe") }
    static var webUI: URL { helperDir.appendingPathComponent("vpipe-web-ui") }

    // A missing helper is a broken bundle, not a user error -- report it
    // once at the top rather than as a spawn failure per action.
    static var missingHelpers: [String] {
        var out: [String] = []
        if !FileManager.default.isExecutableFile(atPath: cli.path) {
            out.append(cli.path)
        }
        if !FileManager.default.isExecutableFile(atPath: webUI.path) {
            out.append(webUI.path)
        }
        return out
    }
}

enum LogLevel: String, CaseIterable, Identifiable {
    case error, warn, info, normal, verbose, debug
    var id: String { rawValue }

    // Display only. The rawValue is what goes into the session config,
    // and parse_log_level (interfaces/log-delegate-intf.cc) matches
    // LOWERCASE names exactly -- an unknown string silently falls back
    // to the default rather than erroring, so capitalizing the value
    // itself would quietly ignore the setting.
    var display: String {
        switch self {
        case .error:   return "Error"
        case .warn:    return "Warn"
        case .info:    return "Info"
        case .normal:  return "Normal"
        case .verbose: return "Verbose"
        case .debug:   return "Debug"
        }
    }
}

// The session config the helpers are launched with, plus the work
// directory that is not part of that config but decides everything
// about where a run puts its state.
//
// Persisted in UserDefaults rather than a file: it is small, it is
// per-user, and it must survive the app being replaced by an update,
// which a file inside the bundle would not.
@MainActor
final class AppModel: ObservableObject {
    static let shared = AppModel()

    // The locales the session has translations for, mirroring
    // supported_languages() in common/i18n.cc. The empty tag omits the
    // key entirely, leaving the session on its own default.
    static let languages: [(tag: String, name: String)] = [
        ("",      "System Default"),
        ("en-us", "English"),
        ("zh-cn", "中文（简体）"),
        ("zh-tw", "中文（繁體）"),
    ]

    // The directory the helpers are launched FROM. vpipe treats its
    // working directory as its workspace -- models/, the LMDB pair, and
    // (under the web UI) sandbox/ are all created there -- so this one
    // setting decides which volume needs the free space and which
    // registry a prepared model is recorded in.
    @Published var workDir: String {
        didSet { defaults.set(workDir, forKey: "workDir") }
    }
    @Published var logLevel: LogLevel {
        didSet { defaults.set(logLevel.rawValue, forKey: "logLevel") }
    }
    @Published var dbMapSizeMB: Int {
        didSet { defaults.set(dbMapSizeMB, forKey: "dbMapSizeMB") }
    }
    @Published var numWorkers: Int {
        didSet { defaults.set(numWorkers, forKey: "numWorkers") }
    }
    @Published var defaultEdgeCapacity: Int {
        didSet { defaults.set(defaultEdgeCapacity, forKey: "edgeCapacity") }
    }
    @Published var memoryCapMB: Int {
        didSet { defaults.set(memoryCapMB, forKey: "memoryCapMB") }
    }
    @Published var language: String {
        didSet { defaults.set(language, forKey: "language") }
    }
    @Published var plugins: [String] {
        didSet { defaults.set(plugins, forKey: "plugins") }
    }

    // Directory to load FFmpeg from, or empty for the built-in search
    // order (bundled copy first, then the system install).
    //
    // Worth exposing because the bundled build is deliberately minimal:
    // LGPL only, so no libx264/x265 and none of the many decoders a
    // Homebrew build carries. Someone who has installed a fuller FFmpeg
    // should be able to use it. And for a build that bundles NOTHING,
    // this is the only way to point at an install the loader's fixed
    // candidate list does not already know about.
    @Published var ffmpegDir: String {
        didSet { defaults.set(ffmpegDir, forKey: "ffmpegDir") }
    }

    // Filesystem access for the web UI.
    //
    // By default stage file I/O is confined to <work>/sandbox.
    // exposeNativeFS drops that entirely; whitelistPaths punches
    // specific real directories through it. The two are mutually
    // exclusive by the CLI's own rule -- --white-list-path is ignored
    // once --expose-native-file-system is given -- so the UI disables
    // them rather than letting someone set a whitelist that does
    // nothing.
    @Published var exposeNativeFS: Bool {
        didSet { defaults.set(exposeNativeFS, forKey: "exposeNativeFS") }
    }
    @Published var whitelistPaths: [String] {
        didSet { defaults.set(whitelistPaths, forKey: "whitelistPaths") }
    }
    // A whole-process seatbelt UNDER the app-level sandbox. Off by
    // default because it cannot be nested, and turning it on disables
    // the run_python chat tool, which sandboxes each call itself.
    @Published var osSandbox: Bool {
        didSet { defaults.set(osSandbox, forKey: "osSandbox") }
    }

    // Web UI launch options.
    //
    // The address to bind the server to. "" means let vpipe-web-ui
    // choose (it picks this Mac's LAN address); "127.0.0.1" restricts
    // it to this machine; anything else is a specific interface the
    // user picked.
    //
    // An explicit address rather than a reachable/not-reachable toggle,
    // because a Mac routinely has several: Wi-Fi, Ethernet, a VPN tap, a
    // virtualisation bridge. "Reachable from other devices" cannot say
    // WHICH others, and serving a control surface on a VPN or a bridge
    // to every container on the box is not the same decision as serving
    // it on the home network.
    @Published var bindAddress: String {
        didSet { defaults.set(bindAddress, forKey: "bindAddress") }
    }
    @Published var port: Int {
        didSet { defaults.set(port, forKey: "port") }
    }
    @Published var useTLS: Bool {
        didSet { defaults.set(useTLS, forKey: "useTLS") }
    }

    private let defaults = UserDefaults.standard

    private init() {
        let d = UserDefaults.standard
        // Default work directory: ~/vpipe. Not the app bundle and not a
        // temp dir -- a 45 GB prepared model has to live somewhere the
        // user can find and an update will not touch.
        workDir = d.string(forKey: "workDir")
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("vpipe").path
        logLevel = LogLevel(rawValue: d.string(forKey: "logLevel") ?? "normal")
            ?? .normal
        dbMapSizeMB = d.object(forKey: "dbMapSizeMB") as? Int ?? 0
        numWorkers = d.object(forKey: "numWorkers") as? Int ?? 0
        defaultEdgeCapacity = d.object(forKey: "edgeCapacity") as? Int ?? 0
        memoryCapMB = d.object(forKey: "memoryCapMB") as? Int ?? 0
        language = d.string(forKey: "language") ?? ""
        plugins = d.stringArray(forKey: "plugins") ?? []
        ffmpegDir = d.string(forKey: "ffmpegDir") ?? ""
        exposeNativeFS = d.bool(forKey: "exposeNativeFS")
        whitelistPaths = d.stringArray(forKey: "whitelistPaths") ?? []
        osSandbox = d.bool(forKey: "osSandbox")
        // Migrate the old boolean: false meant loopback-only.
        if let a = d.string(forKey: "bindAddress") {
            bindAddress = a
        } else if let legacy = d.object(forKey: "bindLAN") as? Bool,
                  legacy == false {
            bindAddress = "127.0.0.1"
        } else {
            bindAddress = ""
        }
        port = d.object(forKey: "port") as? Int ?? 9876
        useTLS = d.bool(forKey: "useTLS")
    }

    var workDirURL: URL { URL(fileURLWithPath: workDir) }

    // vpipe-web-ui confines stage file I/O here (its --config
    // file_sandbox.root), and creates it at startup. It may not exist
    // until the server has run once.
    var sandboxURL: URL { workDirURL.appendingPathComponent("sandbox") }

    // True when the server will answer to something other than this
    // machine. Empty means vpipe-web-ui picks the LAN address, so that
    // counts as exposed too.
    var bindIsRemote: Bool {
        bindAddress != "127.0.0.1" && bindAddress != "localhost"
    }

    // Environment for every helper launch. VPIPE_FFMPEG_DIR is read by
    // common/ffmpeg-libraries.cc and tried BEFORE the bundled copy and
    // the system paths.
    var helperEnvironment: [String: String] {
        ffmpegDir.isEmpty ? [:] : ["VPIPE_FFMPEG_DIR": ffmpegDir]
    }

    // Asked of the shipped binary rather than read from Info.plist.
    //
    // `vpipe --version` returns vpipe_version(), i.e. the version and
    // the GIT REVISION compiled into the libvpipe inside this bundle
    // ("0.1 (f913cf3*10)", where *N counts files dirty at build time).
    // That is the identity worth showing: it names the code that will
    // actually run, and it is what a bug report needs. CFBundleVersion
    // stays a plain commit count because Sparkle requires a
    // monotonically increasing integer there -- it identifies the
    // RELEASE, not the source.
    //
    // Cached: this spawns a process, and the answer cannot change while
    // the app is running.
    private static var cachedVersion: String?

    static var versionString: String {
        if let v = cachedVersion { return v }
        var out = "unknown"
        let p = Process()
        p.executableURL = BundlePaths.cli
        p.arguments = ["--version"]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = Pipe()
        if (try? p.run()) != nil {
            let d = pipe.fileHandleForReading.readDataToEndOfFile()
            p.waitUntilExit()
            let s = String(decoding: d, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !s.isEmpty { out = s }
        }
        cachedVersion = out
        return out
    }

    // The six libraries vpipe dlopens, in the unversioned spelling the
    // loader asks for first.
    static let ffmpegLibNames = ["libavutil", "libavcodec", "libavformat",
                                 "libavdevice", "libswresample", "libswscale"]

    // What a directory actually offers, for the Settings readout.
    //
    // Worth checking in the UI because the failure is otherwise SILENT:
    // the loader probes each candidate with dlopen and treats a failure
    // as "not present", so a wrong or incomplete directory does not
    // report an error -- it quietly falls back to a different FFmpeg.
    // That cost real debugging time during bring-up; a user pointing
    // this at the wrong folder deserves to be told immediately.
    static func inspectFFmpegDir(_ path: String) -> (ok: Bool, detail: String) {
        if path.isEmpty {
            return (true, "Using the default search order: the copy inside "
                        + "the app if it has one, otherwise the system "
                        + "install.")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir),
              isDir.boolValue else {
            return (false, "This is not a directory.")
        }
        let missing = ffmpegLibNames.filter {
            !FileManager.default.fileExists(
                atPath: path + "/" + $0 + ".dylib")
        }
        if missing.isEmpty {
            return (true, "All six libraries found.")
        }
        if missing.count == ffmpegLibNames.count {
            return (false, "No FFmpeg libraries here. Point at the lib/ "
                         + "directory of an FFmpeg install, e.g. "
                         + "/opt/homebrew/lib.")
        }
        return (false, "Missing: " + missing.joined(separator: ", ")
                     + ". vpipe would silently fall back to another "
                     + "FFmpeg rather than report an error.")
    }

    func ensureWorkDir() throws {
        try FileManager.default.createDirectory(
            at: workDirURL, withIntermediateDirectories: true)
    }

    // The session config as inline JSON for --config.
    //
    // Zero means "unset" for every numeric field, and an unset field is
    // OMITTED rather than sent as 0. The helpers each have a considered
    // default (worker count from the core count, edge capacity 4,
    // uncapped memory); writing a 0 over one of those is not the same as
    // staying quiet, and for num_workers it would be clamped to 1 -- a
    // single-threaded pipeline nobody asked for.
    func sessionConfigJSON() -> String {
        var root: [String: Any] = [:]

        // db.path defaults to the process CWD, which is already the work
        // directory, so it is only worth sending when the map size is.
        if dbMapSizeMB > 0 {
            root["db"] = ["map_size_mb": dbMapSizeMB]
        }
        root["log"] = ["level": logLevel.rawValue]
        if numWorkers > 0 { root["pool"] = ["num_workers": numWorkers] }
        if defaultEdgeCapacity > 0 {
            root["pipeline"] = ["default_edge_capacity": defaultEdgeCapacity]
        }
        if memoryCapMB > 0 { root["memory_cap_mb"] = memoryCapMB }
        if !language.isEmpty { root["language"] = language }
        if !plugins.isEmpty { root["plugins"] = plugins }

        guard let data = try? JSONSerialization.data(withJSONObject: root),
              let s = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return s
    }
}
