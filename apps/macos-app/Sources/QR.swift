import AppKit
import CoreImage
import CoreImage.CIFilterBuiltins

// The connection QR, rendered natively.
//
// vpipe-web-ui already draws one on its console, but that is a block of
// terminal escape codes -- fine in a terminal, useless in a window. The
// link itself comes from --emit-connection-json, so the app encodes it
// rather than trying to recover pixels from the console rendering.
enum QR {
    static func image(for string: String, scale: CGFloat = 8) -> NSImage? {
        let filter = CIFilter.qrCodeGenerator()
        filter.message = Data(string.utf8)
        // Medium correction: the link is short and the code will be
        // scanned off a clean screen, so spending redundancy on damage
        // tolerance would only make the modules smaller.
        filter.correctionLevel = "M"
        guard let out = filter.outputImage else { return nil }

        // Scale by integer nearest-neighbour BEFORE rasterising. A QR is
        // a hard-edged bitmap; letting the view smooth it on the way up
        // is what makes a code that will not scan.
        let big = out.transformed(by: CGAffineTransform(scaleX: scale,
                                                        y: scale))
        let ctx = CIContext()
        guard let cg = ctx.createCGImage(big, from: big.extent) else {
            return nil
        }
        return NSImage(cgImage: cg,
                       size: NSSize(width: big.extent.width,
                                    height: big.extent.height))
    }
}
