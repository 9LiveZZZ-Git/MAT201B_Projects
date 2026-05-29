#!/usr/bin/env python3
"""Write / verify corvid assets/crows/student.bin.

This is the FORMAT AUTHORITY for the distilled crow-splat student consumed by
viz/SplatModel.cpp (runtime, hand-rolled, no ML deps). Offline only.

Layout (little-endian), per SplatModel.cpp:
    magic   char[4] = "CRVS"
    version int32   = 1
    d_in    int32   = 11        (= 3 base_pos + 8 cond)
    d_hidden int32  = 16..4096
    d_out   int32   = 7         (= 3 dpos + 3 dcol + 1 dsigma)
    W1 float32[d_hidden*d_in]  row-major [d_hidden][d_in]
    b1 float32[d_hidden]
    W2 float32[d_out*d_hidden] row-major [d_out][d_hidden]
    b2 float32[d_out]

Usage:
    # smoke: random small student so the C++ loader path can be exercised
    python3 export_student.py --random --d-hidden 64 --out ../../assets/crows/student.bin

    # real: export trained weights from an .npz with arrays W1,b1,W2,b2
    python3 export_student.py --weights student.npz --out ../../assets/crows/student.bin

    # verify an existing file round-trips and matches the contract
    python3 export_student.py --verify ../../assets/crows/student.bin
"""
import argparse
import struct
import sys

MAGIC = b"CRVS"
VERSION = 1
D_IN = 11      # 3 + COND(8)  — must match SplatModel.hpp COND
D_OUT = 7      # dpos3 + dcol3 + dsigma1


def _write(path, d_hidden, W1, b1, W2, b2):
    assert len(W1) == d_hidden * D_IN, "W1 size"
    assert len(b1) == d_hidden, "b1 size"
    assert len(W2) == D_OUT * d_hidden, "W2 size"
    assert len(b2) == D_OUT, "b2 size"
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<iiii", VERSION, D_IN, d_hidden, D_OUT))
        f.write(struct.pack("<%df" % len(W1), *W1))
        f.write(struct.pack("<%df" % len(b1), *b1))
        f.write(struct.pack("<%df" % len(W2), *W2))
        f.write(struct.pack("<%df" % len(b2), *b2))
    print("wrote %s  (d_hidden=%d, %d floats)" %
          (path, d_hidden, len(W1) + len(b1) + len(W2) + len(b2)))


def cmd_random(args):
    import random
    random.seed(args.seed)
    dh = args.d_hidden
    # small weights -> tiny deltas at init (cloud stays near the seeded crow form)
    s = 0.05
    rnd = lambda n: [random.uniform(-s, s) for _ in range(n)]
    _write(args.out, dh, rnd(dh * D_IN), [0.0] * dh, rnd(D_OUT * dh), [0.0] * D_OUT)


def cmd_weights(args):
    try:
        import numpy as np
    except ImportError:
        sys.exit("--weights needs numpy (offline only): pip install numpy")
    z = np.load(args.weights)
    W1, b1, W2, b2 = z["W1"], z["b1"], z["W2"], z["b2"]
    dh = b1.shape[0]
    if W1.shape != (dh, D_IN):
        sys.exit("W1 must be [d_hidden, %d], got %s" % (D_IN, W1.shape))
    if W2.shape != (D_OUT, dh):
        sys.exit("W2 must be [%d, d_hidden], got %s" % (D_OUT, W2.shape))
    _write(args.out, dh,
           W1.astype("float32").ravel().tolist(),
           b1.astype("float32").ravel().tolist(),
           W2.astype("float32").ravel().tolist(),
           b2.astype("float32").ravel().tolist())


def cmd_verify(args):
    with open(args.verify, "rb") as f:
        data = f.read()
    if data[:4] != MAGIC:
        sys.exit("bad magic %r (want %r)" % (data[:4], MAGIC))
    ver, din, dh, dout = struct.unpack_from("<iiii", data, 4)
    if din != D_IN or dout != D_OUT or not (0 < dh <= 4096):
        sys.exit("bad header: version=%d d_in=%d d_hidden=%d d_out=%d" % (ver, din, dh, dout))
    need = 20 + 4 * (dh * din + dh + dout * dh + dout)
    if len(data) != need:
        sys.exit("size mismatch: have %d bytes, expected %d" % (len(data), need))
    print("OK: version=%d d_in=%d d_hidden=%d d_out=%d size=%d bytes" %
          (ver, din, dh, dout, len(data)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="student.bin", help="output path")
    p.add_argument("--random", action="store_true", help="emit a random smoke student")
    p.add_argument("--d-hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weights", help=".npz with W1,b1,W2,b2")
    p.add_argument("--verify", help="verify an existing student.bin and exit")
    a = p.parse_args()
    if a.verify:
        cmd_verify(a)
    elif a.weights:
        cmd_weights(a)
    elif a.random:
        cmd_random(a)
    else:
        p.error("pick one of --random, --weights, or --verify")


if __name__ == "__main__":
    main()
