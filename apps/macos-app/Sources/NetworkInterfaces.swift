import Foundation

// The IPv4 addresses this Mac can serve the web UI on.
//
// Enumerated rather than offered as free text: an address that is not
// actually assigned to an interface makes bind() fail, and the failure
// arrives as a dead server rather than as a message about the address.
struct NetInterface: Identifiable, Hashable {
    let name: String       // BSD name, e.g. "en0"
    let address: String    // dotted IPv4
    let isLoopback: Bool
    var id: String { address }

    // en0 is Wi-Fi/Ethernet on essentially every Mac; the rest are
    // usually bridges, VPNs and virtualisation taps that nobody means
    // to serve on.
    var label: String {
        if isLoopback { return "This Mac only (\(address))" }
        return "\(name) — \(address)"
    }
}

enum NetworkInterfaces {
    static func ipv4() -> [NetInterface] {
        var out: [NetInterface] = []
        var head: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&head) == 0, let first = head else { return out }
        defer { freeifaddrs(head) }

        for ptr in sequence(first: first, next: { $0.pointee.ifa_next }) {
            let ifa = ptr.pointee
            guard let sa = ifa.ifa_addr,
                  sa.pointee.sa_family == UInt8(AF_INET) else { continue }
            let flags = Int32(ifa.ifa_flags)
            guard flags & IFF_UP != 0 else { continue }

            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            let rc = getnameinfo(sa, socklen_t(sa.pointee.sa_len),
                                 &host, socklen_t(host.count),
                                 nil, 0, NI_NUMERICHOST)
            guard rc == 0 else { continue }
            let addr = String(cString: host)
            guard !addr.isEmpty else { continue }

            out.append(NetInterface(
                name: String(cString: ifa.ifa_name),
                address: addr,
                isLoopback: flags & IFF_LOOPBACK != 0))
        }
        // Stable order, loopback last: the useful choices come first.
        return out.sorted {
            $0.isLoopback != $1.isLoopback ? !$0.isLoopback
                                           : $0.name < $1.name
        }
    }
}
