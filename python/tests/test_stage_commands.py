"""Stage commands from Python: data in, data out, in place.

Run against a build tree (no third-party packages needed -- the stdlib
`array` / `memoryview` stand in for numpy, which works the same way):

    PYTHONPATH=<build>/python python3 python/tests/test_stage_commands.py

in (external-input) -> tap (external-tap, gate) -> end (external-tap, gate)
"""

import array
import json
import sys
import time

import vpipe

GRAPH = {
    "id": "py-commands",
    "stages": [
        {"id": "in", "type": "external-input", "iports": [], "config": {}},
        {"id": "tap", "type": "external-tap",
         "iports": [{"src": "in", "oport": 0}], "config": {"mode": "gate"}},
        {"id": "end", "type": "external-tap",
         "iports": [{"src": "tap", "oport": 0}], "config": {"mode": "gate"}},
    ],
    "subpipelines": [],
}


def f32(values, shape):
    """A float32 buffer of the given shape, as a plain memoryview."""
    return memoryview(array.array("f", values)).cast("B").cast("f", shape)


def main():
    s = vpipe.session
    p = s.load_pipeline(json.dumps(GRAPH))
    assert p, "load_pipeline failed"
    src, tap = p.stage("in"), p.stage("tap")
    assert src and tap and src.id == "in"
    names = [c["name"] for c in src.commands()]
    assert names == ["push", "lease", "finish"], names

    # Refused before launch -- as a value, not an exception.
    c = tap.command("read")
    assert c.state == "failed" and "not running" in c.error, c.error

    assert s.launch_pipeline(p).code == 0
    try:
        # IN: the caller's memory, read in place by the stage.
        r = src.call("push", args={"sideband": {"frame": 1}},
                     buffers={"data": f32([1, 2, 3, 4], [2, 2])},
                     timeout_ms=3000)
        assert r == {"seq": 1}, r

        # OUT: the beat's own bytes, writable, held until close.
        with tap.command("read") as rd:
            assert rd.wait(3000) == "replied", rd.error
            assert rd.result["sideband"] == {"frame": 1}
            view = memoryview(rd.buffers["data"])
            assert view.format == "f" and view.shape == (2, 2)
            assert view.tolist() == [[1.0, 2.0], [3.0, 4.0]]
            view[0, 0] = 9.0          # an edit that travels downstream
        # A stage that holds is released by the `with` block's close.

        # The edit reached the next stage, which holds it in turn.
        end = p.stage("end")
        with end.command("read") as e1:
            assert e1.wait(3000) == "replied", e1.error
            assert memoryview(e1.buffers["data"]).tolist()[0] == [9.0, 2.0]

        # A strided (non-contiguous) input is gathered by the stage.
        every_other = memoryview(array.array("f", [1, 0, 2, 0, 3, 0]))[::2]
        assert not every_other.contiguous
        src.call("push", buffers={"data": every_other}, timeout_ms=3000)
        for stage in (tap, end):
            with stage.command("read") as rd:
                assert rd.wait(3000) == "replied"
                got = memoryview(rd.buffers["data"]).tolist()
        assert got == [1.0, 2.0, 3.0], got

        # LEASE: fill the stage's own storage, emit it on close.
        with src.command("lease",
                         args={"type": "u8", "shape": [3]}) as ls:
            assert ls.wait(3000) == "replied", ls.error
            buf = memoryview(ls.buffers["data"])
            assert not buf.readonly
            buf[:] = bytes([7, 8, 9])
        for stage in (tap, end):
            with stage.command("read") as rd3:
                assert rd3.wait(3000) == "replied"
                got = bytes(memoryview(rd3.buffers["data"]))
        assert got == b"\x07\x08\x09", got

        # Many quick round trips: the caller's buffers are dropped on
        # vpipe's threads as often as on this one.
        tap_mode_reads = []
        for i in range(200):
            src.call("push", buffers={"data": bytearray([i % 256])},
                     timeout_ms=3000)
            for stage in (tap, end):
                with stage.command("read") as rd:
                    assert rd.wait(3000) == "replied"
                    tap_mode_reads.append(rd.result["seq"])
        assert tap_mode_reads[-1] > 200

        # Bytes go in as-is, and a bad argument is named.
        bad = src.command("push", args={"sidebnad": {}},
                          buffers={"data": b"abc"})
        assert bad.state == "failed" and "sidebnad" in bad.error, bad.error
        try:
            src.call("lease", args={"type": "f64", "shape": [1]},
                     timeout_ms=3000)
            raise AssertionError("f64 lease should fail")
        except RuntimeError as e:
            assert "f64" in str(e), e
    finally:
        assert s.stop_pipeline(p).code == 0
        s.unload_pipeline(p)
    print("stage commands from Python: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
