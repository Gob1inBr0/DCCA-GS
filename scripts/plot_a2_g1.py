"""A2 (submodular v1 cover) G1 figure — scene 1-78, 30k, r_post=0.85.

Left: same-size PSNR gain of A2 over base, per lambda per seed (the G1 gate
view: gate +0.15 dB, noise band +-0.1 dB).
Right: where A2 sits on the rate-distortion plane vs base (both seeds).

Numbers from docs/data/experiments.csv (groups ph0a/ph0c/ph1v3/jr2) as
registered through 2026-09-17; lambda 0.0005 seed 104 still pending.
Same-size deltas follow the CSV notes / group-report 2026-09-17:
  lam0004: s42 +0.521 (base 26.9829@11.084), s104 +0.422 (same-size ref 26.983)
  lam0002: s42 +0.076 (base 27.7298@14.453), s104 clean rerun -0.019
           (new-code base 27.695@13.90)
  lam0005: s42 +0.122 (base 27.5974@21.441, -0.89 MB), s104 pending
"""

from __future__ import annotations

# (lambda label, tight->loose) same-size deltas per seed
GAIN_S42 = [0.521, 0.076, 0.122]
GAIN_S104 = [0.422, -0.019, None]  # None = run still pending
LAMS = ["0.004\n(tight)", "0.002", "0.0005\n(loose)"]

BASE_S42 = [  # (MB, PSNR) r085, lambda 0.004 / 0.002 / 0.0005
    (11.0835, 26.9829),
    (14.4530, 27.7298),
    (21.4410, 27.5974),
]
BASE_S104 = [  # lambda 0.002 s104 base not registered
    (11.7290, 27.2617),
    (20.9158, 27.4487),
]
A2_S42 = [
    (11.0557, 27.5038),
    (14.4323, 27.8058),
    (20.5475, 27.7186),
]
A2_S104 = [  # lambda 0.002 = clean rerun s104r; 0.0005 pending
    (11.2760, 27.4046),
    (14.4984, 27.6764),
]

GRAY = "#7f7f7f"
RED = "#d62728"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=150)

    # ---- panel 1: same-size gain vs base (G1 view) ----
    xs = range(len(LAMS))
    w = 0.36
    b1 = ax1.bar([x - w / 2 for x in xs], GAIN_S42, width=w, color=RED,
                 alpha=0.85, label="seed 42", zorder=3)
    gain104 = [g if g is not None else 0 for g in GAIN_S104]
    b2 = ax1.bar([x + w / 2 for x in xs], gain104, width=w, color=RED,
                 alpha=0.45, edgecolor=RED, hatch="//", label="seed 104",
                 zorder=3)
    ax1.axhspan(-0.1, 0.1, color="gray", alpha=0.15, zorder=1,
                label="run noise ±0.1 dB (F7)")
    ax1.axhline(0.15, color="black", ls="--", lw=1.2, zorder=2,
                label="G1 gate +0.15 dB")
    ax1.axhline(0, color="black", lw=0.8, zorder=2)
    for x, g in zip(xs, GAIN_S42):
        ax1.text(x - w / 2, g + 0.02, f"+{g:.3f}" if g >= 0 else f"{g:.3f}",
                 ha="center", fontsize=9, color=RED)
    for x, g in zip(xs, GAIN_S104):
        if g is None:
            ax1.text(x + w / 2, 0.05, "pending", ha="center", va="bottom",
                     fontsize=8, style="italic", color="#555555")
        else:
            ax1.text(x + w / 2, g + (0.02 if g >= 0 else -0.06),
                     f"{g:+.3f}", ha="center", fontsize=9, color="#8a3030")
    ax1.text(0.02, 0.965, "λ=0.004 two-seed mean +0.47 dB\n"
             "5-value mean +0.22 dB (4 pos / 1 flat)",
             transform=ax1.transAxes, ha="left", va="top", fontsize=9,
             bbox=dict(fc="white", ec="#cccccc", alpha=0.9))
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(LAMS)
    ax1.set_ylim(-0.18, 0.68)
    ax1.set_xlabel("rate penalty λ (pruning deepest at 0.004)")
    ax1.set_ylabel("ΔPSNR vs same-size base (dB)")
    ax1.set_title("A2 submodular cover — same-size gain over base\n"
                  "(scene 1-78, 30k, r_post=0.85)")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.5)

    # ---- panel 2: RD plane, base vs A2 ----
    ax2.plot([p[0] for p in BASE_S42], [p[1] for p in BASE_S42], "o-",
             color=GRAY, ms=7, lw=1.3, label="base s42", zorder=3)
    ax2.plot([p[0] for p in BASE_S104], [p[1] for p in BASE_S104], "o--",
             color=GRAY, ms=7, lw=1.1, label="base s104", zorder=3)
    ax2.plot([p[0] for p in A2_S42], [p[1] for p in A2_S42], "D-",
             color=RED, ms=7.5, lw=1.5, label="A2 s42", zorder=4)
    ax2.plot([p[0] for p in A2_S104], [p[1] for p in A2_S104], "D--",
             color=RED, ms=7, lw=1.2, label="A2 s104 (0.002 = clean rerun)",
             zorder=4)
    # same-lambda shift arrows, seed 42 (tight and mid)
    for (bx, by), (ax_, ay), lab in [
        (BASE_S42[0], A2_S42[0], "+0.52"),
        (BASE_S42[1], A2_S42[1], "+0.08"),
        (BASE_S42[2], A2_S42[2], "+0.12\n(−0.89 MB)"),
    ]:
        ax2.annotate("", xy=(ax_, ay), xytext=(bx, by),
                     arrowprops=dict(arrowstyle="->", color="#d6272888",
                                     lw=1.2), zorder=2)
        ax2.text((ax_ + bx) / 2 + 0.28, (ay + by) / 2, lab, fontsize=8.5,
                 color=RED, ha="left", va="center")
    ax2.text(0.02, 0.97, "λ=0.0005 s104 pending\nBD-rate vs HAC++ "
             "(extrapolated):\nbase −25.3% | +A2 −34.2%",
             transform=ax2.transAxes, ha="left", va="top", fontsize=8,
             color="#444444",
             bbox=dict(fc="white", ec="#cccccc", alpha=0.9))
    ax2.set_xlabel("decoded size (MB, fp32 accounting)")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("A2 vs base on the rate-distortion plane\n"
                  "(each λ one point; tight budget gains quality AND size)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    out = "docs/figures/a2_submod_g1_1-78.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
