import argparse
import array
import math
import os
import sys


def read_floats(path: str) -> array.array:
    size = os.path.getsize(path)
    if size % 4 != 0:
        raise ValueError(f"{path}: size {size} is not a multiple of 4 bytes")
    count = size // 4
    data = array.array("f")
    with open(path, "rb") as f:
        data.fromfile(f, count)
    return data


def summarize(
    computed: array.array,
    solution: array.array,
    kind: str,
    atol: float,
    rtol: float,
    max_report: int,
) -> int:
    if len(computed) != len(solution):
        raise ValueError(
            f"{kind}: length mismatch computed={len(computed)} solution={len(solution)}"
        )
    if len(computed) % 3 != 0:
        raise ValueError(f"{kind}: data length {len(computed)} not divisible by 3")

    names = ["x", "y", "z"] if kind == "pos" else ["vx", "vy", "vz"]
    n = len(computed)
    n_bodies = n // 3

    max_abs = -1.0
    max_abs_idx = -1
    max_rel = -1.0
    max_rel_idx = -1
    sum_abs = 0.0
    sum_sq = 0.0
    bad = []
    bad_count = 0
    nan_count = 0
    inf_count = 0

    max_body_l2 = -1.0
    max_body_idx = -1

    for i in range(n_bodies):
        l2 = 0.0
        for c in range(3):
            idx = i * 3 + c
            cv = computed[idx]
            sv = solution[idx]
            if math.isnan(cv) or math.isnan(sv):
                nan_count += 1
                continue
            if math.isinf(cv) or math.isinf(sv):
                inf_count += 1
                continue

            diff = abs(cv - sv)
            l2 += diff * diff
            sum_abs += diff
            sum_sq += diff * diff

            if diff > max_abs:
                max_abs = diff
                max_abs_idx = idx

            denom = abs(sv) if abs(sv) > 0.0 else 1.0
            rel = diff / denom
            if rel > max_rel:
                max_rel = rel
                max_rel_idx = idx

            tol = atol + rtol * abs(sv)
            if diff > tol:
                bad_count += 1
                if len(bad) < max_report:
                    bad.append((i, names[c], cv, sv, diff, rel, tol))

        l2 = math.sqrt(l2)
        if l2 > max_body_l2:
            max_body_l2 = l2
            max_body_idx = i

    mean_abs = sum_abs / n if n else 0.0
    rms = math.sqrt(sum_sq / n) if n else 0.0
    bad_pct = 100.0 * bad_count / n if n else 0.0

    print(f"{kind}: bodies={n_bodies} components={n}")
    print(f"  tol: atol={atol:g} rtol={rtol:g}")
    print(f"  max_abs={max_abs:g} max_rel={max_rel:g}")
    print(f"  mean_abs={mean_abs:g} rms={rms:g}")
    print(f"  max_body_l2={max_body_l2:g} at body={max_body_idx}")
    if nan_count or inf_count:
        print(f"  NaN={nan_count} Inf={inf_count}")
    print(f"  above_tol={bad_count} ({bad_pct:.4f}%)")

    if bad:
        print("  worst samples (body, comp, computed, solution, abs, rel, tol):")
        for item in bad:
            body, comp, cv, sv, diff, rel, tol = item
            print(
                f"    {body:6d} {comp:>2}  {cv:.9g}  {sv:.9g}  "
                f"{diff:.3g}  {rel:.3g}  {tol:.3g}"
            )

    return 0 if bad_count == 0 and nan_count == 0 and inf_count == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare computed vs solution N-body binaries with tolerance."
    )
    parser.add_argument("--computed", required=True)
    parser.add_argument("--solution", required=True)
    parser.add_argument("--kind", choices=["pos", "vel"], required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--max-report", type=int, default=10)
    args = parser.parse_args()

    computed = read_floats(args.computed)
    solution = read_floats(args.solution)
    return summarize(
        computed, solution, args.kind, args.atol, args.rtol, args.max_report
    )


if __name__ == "__main__":
    sys.exit(main())
