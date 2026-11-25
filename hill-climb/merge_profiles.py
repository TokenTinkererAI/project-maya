"""Merge IK2_POP_OUT profiles (int32 n_layer, int32 E, uint64[n_layer*E] pick counts).

Usage: python merge_profiles.py OUT IN1 [IN2 ...]
Counts are summed element-wise; headers must match.
"""
import struct
import sys


def read_profile(path):
    with open(path, "rb") as f:
        hdr = f.read(8)
        n_layer, e = struct.unpack("<ii", hdr)
        n = n_layer * e
        data = f.read(8 * n)
        if len(data) != 8 * n:
            raise SystemExit(f"{path}: truncated (want {8*n} bytes of counts, got {len(data)})")
        counts = struct.unpack(f"<{n}Q", data)
    return n_layer, e, counts


def main():
    out_path, inputs = sys.argv[1], sys.argv[2:]
    if not inputs:
        raise SystemExit("no input profiles given")
    n_layer, e, total = read_profile(inputs[0])
    for p in inputs[1:]:
        nl2, e2, c = read_profile(p)
        if (nl2, e2) != (n_layer, e):
            raise SystemExit(f"{p}: header mismatch ({nl2},{e2}) vs ({n_layer},{e})")
        total = tuple(a + b for a, b in zip(total, c))
    with open(out_path, "wb") as f:
        f.write(struct.pack("<ii", n_layer, e))
        f.write(struct.pack(f"<{n_layer * e}Q", *total))
    print(f"wrote {out_path}: n_layer={n_layer} E={e} total_picks={sum(total)} from {len(inputs)} profiles")


if __name__ == "__main__":
    main()
