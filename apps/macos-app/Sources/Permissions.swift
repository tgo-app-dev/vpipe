import AVFoundation
import AppKit
import Foundation
import Network

enum PermState { case granted, denied, undetermined, unknown }

struct PermissionRow: Identifiable {
    let id: String
    let title: String
    let detail: String
    let state: PermState
    // Nil when the system offers no way to ask -- Full Disk Access is
    // grant-only through System Settings, and macOS provides no API to
    // request it.
    let request: (() async -> Void)?
    let settingsURL: String?
}

// The startup checklist, as something the user can act on.
//
// vpipe-web-ui already reports these, but run from a terminal every one
// of them is attributed to the TERMINAL: the checklist describes
// Terminal.app's permissions, granting them means finding Terminal in
// System Settings, and under tmux or ssh it is some other process again.
// Inside a signed bundle the app is its own TCC subject, so the same
// checks describe the app, the prompts name the app, and the helpers it
// spawns inherit that identity.
@MainActor
final class Permissions: ObservableObject {
    @Published private(set) var rows: [PermissionRow] = []
    @Published private(set) var localNetworkProbed = false

    private var browser: NWBrowser?

    init() { refresh() }

    func refresh() {
        rows = [cameraRow(), microphoneRow(), localNetworkRow(),
                fullDiskRow()]
    }

    // MARK: camera / microphone

    private func avState(_ media: AVMediaType) -> PermState {
        switch AVCaptureDevice.authorizationStatus(for: media) {
        case .authorized:    return .granted
        case .denied:        return .denied
        case .restricted:    return .denied
        case .notDetermined: return .undetermined
        @unknown default:    return .unknown
        }
    }

    private func cameraRow() -> PermissionRow {
        let s = avState(.video)
        return PermissionRow(
            id: "camera",
            title: "Camera",
            detail: describe(s, when: "video-capture stages and live "
                                    + "vision-model input"),
            state: s,
            request: s == .undetermined ? { [weak self] in
                _ = await AVCaptureDevice.requestAccess(for: .video)
                await MainActor.run { self?.refresh() }
            } : nil,
            settingsURL: "x-apple.systempreferences:com.apple.preference"
                       + ".security?Privacy_Camera")
    }

    private func microphoneRow() -> PermissionRow {
        let s = avState(.audio)
        return PermissionRow(
            id: "microphone",
            title: "Microphone",
            detail: describe(s, when: "audio capture, transcription and "
                                    + "audio tagging"),
            state: s,
            request: s == .undetermined ? { [weak self] in
                _ = await AVCaptureDevice.requestAccess(for: .audio)
                await MainActor.run { self?.refresh() }
            } : nil,
            settingsURL: "x-apple.systempreferences:com.apple.preference"
                       + ".security?Privacy_Microphone")
    }

    // MARK: local network

    // There is no authorization-status API for Local Network the way
    // there is for camera and microphone, so this cannot be READ -- it
    // can only be exercised. Starting a Bonjour browse is what triggers
    // the system prompt; whether anything is found afterwards is a weak
    // signal (a quiet network has no responders either), so the row
    // reports that the probe ran rather than claiming a verdict.
    private func localNetworkRow() -> PermissionRow {
        PermissionRow(
            id: "localnetwork",
            title: "Local Network",
            detail: localNetworkProbed
                ? "Probe sent. ONVIF camera discovery and RTSP streams "
                + "need this; without it they find nothing."
                : "Needed to discover ONVIF cameras, reach RTSP streams, "
                + "and serve the web UI to your phone.",
            state: localNetworkProbed ? .unknown : .undetermined,
            request: { [weak self] in await self?.probeLocalNetwork() },
            settingsURL: "x-apple.systempreferences:com.apple.preference"
                       + ".security?Privacy_LocalNetwork")
    }

    private func probeLocalNetwork() async {
        browser?.cancel()
        let params = NWParameters()
        params.includePeerToPeer = true
        let b = NWBrowser(
            for: .bonjour(type: "_http._tcp", domain: nil), using: params)
        b.stateUpdateHandler = { _ in }
        b.browseResultsChangedHandler = { _, _ in }
        b.start(queue: .main)
        browser = b

        // Long enough for the prompt to appear and a responder to
        // answer; short enough that the button does not feel stuck.
        try? await Task.sleep(nanoseconds: 3_000_000_000)
        b.cancel()
        browser = nil
        localNetworkProbed = true
        refresh()
    }

    // MARK: full disk access

    // Probed by reading a file that is protected but present on every
    // install. macOS offers no request API for this one, so the row can
    // only point at System Settings.
    private func fullDiskRow() -> PermissionRow {
        let probe = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Safari/Bookmarks.plist")
        var s: PermState = .unknown
        if FileManager.default.fileExists(atPath: probe.path) {
            s = (try? Data(contentsOf: probe)) != nil ? .granted : .denied
        }
        return PermissionRow(
            id: "fulldisk",
            title: "Full Disk Access",
            detail: s == .granted
                ? "Granted."
                : "Optional. Only needed to read models or media from "
                + "protected folders such as Desktop, Documents or an "
                + "external volume. macOS has no way for an app to ask "
                + "for this -- it has to be granted in System Settings.",
            state: s,
            request: nil,
            settingsURL: "x-apple.systempreferences:com.apple.preference"
                       + ".security?Privacy_AllFiles")
    }

    private func describe(_ s: PermState, when: String) -> String {
        switch s {
        case .granted:      return "Authorized."
        case .denied:       return "Denied. Needed for \(when)."
        case .undetermined: return "Not yet requested. Needed for \(when)."
        case .unknown:      return "Unknown. Needed for \(when)."
        }
    }

    static func openSettings(_ urlString: String) {
        if let u = URL(string: urlString) { NSWorkspace.shared.open(u) }
    }
}
