import Foundation
import SwiftUI

// Is this machine being held back right now?
//
// WHY NOT ProcessInfo.thermalState ALONE. It is the only thermal signal
// macOS offers an app, and on the machine that matters most -- a fanless
// MacBook Air minutes into a generation -- it reads `nominal` while the
// GPU clock sits at roughly two thirds of the chip's ceiling. It is an
// advisory about what an app should voluntarily stop doing, not a report
// of what the hardware is doing. Someone watching steps get slower got
// no explanation from it, which is the whole complaint this answers.
//
// WHAT REPLACES IT. `vpipe --gpu-thermal` samples the GPU's pstate
// residency histogram for a window and reports three things: the clock
// the GPU ran at, the clock the silicon is rated for, and the share of
// the window it was out of idle. Busy AND clamped is throttling; a low
// clock while idle is just DVFS. MEASURED on an M4 Pro: 722 MHz at 6.5%
// active when idle, 1578 MHz at 100% active under a real model run.
//
// The OS state is still read, and still shown when the GPU is too quiet
// to judge -- it is the better answer for a machine that is merely warm
// and doing nothing.
@MainActor
final class ThermalMonitor: ObservableObject {
    // What the helper reported, or nil before the first sample lands.
    struct Reading {
        var verdict: String          // idle | normal | warm | throttled
        var clockMHz: Double?
        var ceilingMHz: Double?
        var activePct: Double?
        var tempC: Double?
    }

    @Published private(set) var osState: ProcessInfo.ThermalState
    @Published private(set) var reading: Reading?

    private var observer: NSObjectProtocol?
    private var pollTask: Task<Void, Never>?

    // Every 8 seconds, each costing one 600 ms sample in a short-lived
    // process that does no GPU work -- IOReport and SMC reads only, so
    // it cannot perturb the generation it is describing. Slow enough to
    // be free, fast enough that a clamp shows up while the user is still
    // wondering why the run got slower.
    private static let pollInterval: Duration = .seconds(8)
    private static let windowMs = 600

    init() {
        osState = ProcessInfo.processInfo.thermalState
        observer = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification,
            object: nil, queue: nil) { [weak self] _ in
                let now = ProcessInfo.processInfo.thermalState
                Task { @MainActor in self?.osState = now }
            }
        start()
    }

    deinit {
        if let o = observer { NotificationCenter.default.removeObserver(o) }
        pollTask?.cancel()
    }

    private func start() {
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                let r = await Self.sample()
                if Task.isCancelled { return }
                await MainActor.run { self?.reading = r }
                try? await Task.sleep(for: Self.pollInterval)
            }
        }
    }

    // Off the main actor: this blocks for the sampling window.
    private nonisolated static func sample() async -> Reading? {
        await withCheckedContinuation { k in
            DispatchQueue.global(qos: .utility).async {
                k.resume(returning: runHelper())
            }
        }
    }

    private nonisolated static func runHelper() -> Reading? {
        let p = Process()
        p.executableURL = BundlePaths.cli
        p.arguments = ["--gpu-thermal", String(windowMs)]
        let out = Pipe()
        p.standardOutput = out
        p.standardError = Pipe()
        guard (try? p.run()) != nil else { return nil }
        let d = out.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        guard p.terminationStatus == 0,
              let obj = try? JSONSerialization.jsonObject(with: d)
                  as? [String: Any],
              let verdict = obj["verdict"] as? String
        else { return nil }
        return Reading(
            verdict: verdict,
            clockMHz:   obj["clock_mhz"]       as? Double,
            ceilingMHz: obj["ceiling_mhz"]     as? Double,
            activePct:  obj["gpu_active_pct"]  as? Double,
            tempC:      obj["temp_c"]          as? Double)
    }

    // MARK: - presentation
    //
    // The GPU verdict wins when it has one, because it is the specific
    // claim; the OS state is the fallback for a quiet machine.

    private var gpuVerdict: String? {
        guard let v = reading?.verdict else { return nil }
        return (v == "throttled" || v == "warm" || v == "normal") ? v : nil
    }

    var label: String {
        switch gpuVerdict {
        case "throttled": return "GPU Throttled"
        case "warm":      return "Warm"
        case "normal":    return "Normal"
        default:          return osLabel
        }
    }

    private var osLabel: String {
        switch osState {
        case .nominal:  return "Normal"
        case .fair:     return "Warm"
        case .serious:  return "Throttling"
        case .critical: return "Critical"
        @unknown default: return "Unknown"
        }
    }

    var isAlerting: Bool {
        gpuVerdict == "throttled" || gpuVerdict == "warm"
            || (gpuVerdict == nil && osState != .nominal)
    }

    var systemImage: String {
        switch gpuVerdict {
        case "throttled": return "gauge.with.dots.needle.33percent"
        case "warm":      return "thermometer.high"
        case "normal":    return "thermometer.low"
        default:
            switch osState {
            case .nominal:  return "thermometer.low"
            case .fair:     return "thermometer.medium"
            case .serious:  return "thermometer.high"
            case .critical: return "thermometer.sun.fill"
            @unknown default: return "thermometer.medium"
            }
        }
    }

    var tint: Color {
        switch gpuVerdict {
        case "throttled": return .orange
        case "warm":      return .yellow
        case "normal":    return .secondary
        default:
            switch osState {
            case .nominal:  return .secondary
            case .fair:     return .yellow
            case .serious:  return .orange
            case .critical: return .red
            @unknown default: return .secondary
            }
        }
    }

    // Shown on hover, and carrying the numbers on purpose: "Throttled" is
    // a claim about the user's hardware, and they should be able to check
    // it rather than take it.
    var help: String {
        guard let r = reading, let v = gpuVerdict else {
            return osHelp + measuring
        }
        var s = ""
        switch v {
        case "throttled":
            s = "The GPU is busy but running below the clock it is rated "
              + "for — steps take longer, and nothing in the pipeline is "
              + "at fault. Improving airflow under the machine helps more "
              + "than anything in software."
        case "warm":
            s = "Running hot but still at full clock. A fanless Mac will "
              + "usually start clamping if the load continues."
        default:
            s = "The GPU is running at its full clock."
        }
        if let c = r.clockMHz, let ceil = r.ceilingMHz {
            s += String(format: "\n\nGPU %.0f of %.0f MHz", c, ceil)
        }
        if let a = r.activePct {
            s += String(format: " · %.0f%% busy", a)
        }
        if let t = r.tempC {
            s += String(format: " · %.0f °C", t)
        }
        return s
    }

    // Before the first sample, and on a machine where the helper cannot
    // read the counters, say which signal is being shown rather than
    // silently presenting the weaker one as if it were the stronger.
    private var measuring: String {
        reading == nil
            ? "\n\nMeasuring the GPU clock…"
            : "\n\nThe GPU is idle, so its clock says nothing about "
            + "throttling; showing the system thermal state instead."
    }

    private var osHelp: String {
        switch osState {
        case .nominal:
            return "Thermals are normal; performance is unrestricted."
        case .fair:
            return "Warming up. Fans (where present) have ramped; a "
                 + "fanless Mac may already be slowing down."
        case .serious:
            return "The system is throttling — generation steps will "
                 + "take longer."
        case .critical:
            return "The system is throttling hard and may suspend work. "
                 + "Let it cool before starting another run."
        @unknown default:
            return "Thermal state unavailable."
        }
    }
}

// The status row along the bottom of the detail column.
//
// Kept deliberately narrow in its layout demands: a NavigationSplitView
// sizes its window from the detail column's IDEAL width, and a status
// row that reports a wide intrinsic size drags the whole window with
// it. The long text lives in .help(), which costs no layout at all.
struct StatusRow: View {
    @ObservedObject var thermal: ThermalMonitor

    // The window's own rounded corner eats into the trailing end, so a
    // trailing-aligned item needs more clearance than the leading one
    // to avoid sitting in the curve.
    private let trailingInset: CGFloat = 22

    var body: some View {
        HStack(spacing: 6) {
            Spacer(minLength: 0)
            Image(systemName: thermal.systemImage)
                .foregroundStyle(thermal.tint)
            Text("Thermal: \(thermal.label)")
                .foregroundStyle(thermal.isAlerting
                                 ? AnyShapeStyle(thermal.tint)
                                 : AnyShapeStyle(.secondary))
        }
        .font(.caption)
        .lineLimit(1)
        .help(thermal.help)
        .padding(.leading, 12)
        .padding(.trailing, trailingInset)
        .padding(.vertical, 5)
        .frame(maxWidth: .infinity, alignment: .trailing)
        .background(.bar)
        .overlay(alignment: .top) {
            Divider()
        }
    }
}
