import Foundation
import SwiftUI

// The system's thermal state, surfaced in the status row.
//
// Worth showing because the machines this app targets throttle in a way
// that looks like a vpipe problem: a MacBook Air is fanless, so minutes
// of sustained sampling heat-soak the chassis and every step after that
// is slower with nothing in the log to explain it. The indicator turns
// "why did this get slow" into a fact the user can see.
//
// Read-only on purpose. macOS exposes no supported way to raise a fan
// (and an Air has none to raise), so the honest offering is the number,
// not a control.
@MainActor
final class ThermalMonitor: ObservableObject {
    @Published private(set) var state: ProcessInfo.ThermalState

    private var observer: NSObjectProtocol?

    init() {
        state = ProcessInfo.processInfo.thermalState
        // The notification is posted on an arbitrary thread, so hop to
        // the main actor before touching a @Published property.
        observer = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification,
            object: nil, queue: nil) { [weak self] _ in
                let now = ProcessInfo.processInfo.thermalState
                Task { @MainActor in self?.state = now }
            }
    }

    deinit {
        if let o = observer { NotificationCenter.default.removeObserver(o) }
    }

    // Apple's own words for these are about what an APP should do, not
    // what the machine is doing, so they do not read well as a label.
    // These say what the user is seeing instead.
    var label: String {
        switch state {
        case .nominal:  return "Normal"
        case .fair:     return "Warm"
        case .serious:  return "Throttling"
        case .critical: return "Critical"
        @unknown default: return "Unknown"
        }
    }

    var systemImage: String {
        switch state {
        case .nominal:  return "thermometer.low"
        case .fair:     return "thermometer.medium"
        case .serious:  return "thermometer.high"
        case .critical: return "thermometer.sun.fill"
        @unknown default: return "thermometer.medium"
        }
    }

    var tint: Color {
        switch state {
        case .nominal:  return .secondary
        case .fair:     return .yellow
        case .serious:  return .orange
        case .critical: return .red
        @unknown default: return .secondary
        }
    }

    // Shown on hover. "Fair" and "Serious" mean nothing on their own,
    // and the actionable part differs by machine: a fanless Air is
    // already at its limit where a Mac with a fan has headroom left.
    var help: String {
        switch state {
        case .nominal:
            return "Thermals are normal; performance is unrestricted."
        case .fair:
            return "Warming up. Fans (where present) have ramped; a "
                 + "fanless Mac may already be slowing down."
        case .serious:
            return "The system is throttling — generation steps will "
                 + "take longer. Improving airflow under the machine "
                 + "helps more than anything in software."
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
                .foregroundStyle(thermal.state == .nominal
                                 ? AnyShapeStyle(.secondary)
                                 : AnyShapeStyle(thermal.tint))
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
