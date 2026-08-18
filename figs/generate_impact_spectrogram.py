"""Generate the aligned observed/expected impact waveform used in Figure 1."""

from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import lfilter

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    output = (
        Path(__file__).resolve().parent
        / "assets"
        / "impact_audio_waveform_v4.png"
    )
    sample_rate = 16_000
    duration = 2.4
    time = np.arange(int(sample_rate * duration)) / sample_rate
    rng = np.random.default_rng(19)

    # Quiet colored room tone for the observed track.
    room = lfilter([1.0], [1.0, -0.94], rng.standard_normal(time.size))
    room = 0.0018 * room / (np.std(room) + 1e-12)
    audio = room.copy()

    # Observed inflated-ball/concrete impact: a sharp broadband attack followed
    # mainly by damped ball-shell modes.  The massive concrete slab contributes
    # little sustained panel ringing.
    # Its onset is aligned to W_g.
    # Fractions 0.365 and 0.696 align the waveform onsets with the projected
    # floor and wall events when the trace spans the full evidence-plate width.
    onset = 0.875
    elapsed = time - onset
    active = elapsed >= 0
    local_time = elapsed[active]
    attack = 0.38 * rng.standard_normal(local_time.size) * np.exp(-76.0 * local_time)

    frequencies = np.array([165.0, 330.0, 720.0, 1500.0, 3100.0])
    decays = np.array([14.0, 20.0, 32.0, 52.0, 78.0])
    amplitudes = np.array([0.26, 0.18, 0.11, 0.060, 0.030])
    ringing = np.zeros_like(local_time)
    for amplitude, frequency, decay in zip(amplitudes, frequencies, decays):
        ringing += (
            amplitude
            * np.exp(-decay * local_time)
            * np.sin(2 * np.pi * frequency * local_time)
        )

    audio[active] += attack + ringing

    # A single random attack peak should not compress the complete visible
    # waveform.  Normalize by a robust event-window peak and clip only the
    # rarest impulses; this enlarges the plotted pressure trace without
    # changing its onset, decay, or modal frequencies.
    impact_window = active & (elapsed <= 0.22)
    robust_peak = np.quantile(np.abs(audio[impact_window]), 0.99)
    audio = np.clip(audio / (robust_peak + 1e-12), -1.0, 1.0)

    # Counterfactual wall impact expected at W_w.  It is deliberately
    # not added to `audio`: teal/dashed rendering marks an obligation, not an
    # observed event.  A fixed masonry wall gives a shorter, duller response
    # with energy concentrated below the glass-like high-frequency regime.
    expected_audio = np.zeros_like(audio)
    wall_onset = 1.669
    wall_elapsed = time - wall_onset
    wall_active = wall_elapsed >= 0
    wall_time = wall_elapsed[wall_active]
    wall_rng = np.random.default_rng(31)
    wall_attack = (
        0.22 * wall_rng.standard_normal(wall_time.size) * np.exp(-74.0 * wall_time)
    )
    wall_frequencies = np.array([140.0, 280.0, 560.0, 1120.0, 2240.0])
    wall_decays = np.array([14.0, 19.0, 28.0, 44.0, 68.0])
    wall_amplitudes = np.array([0.22, 0.17, 0.105, 0.060, 0.028])
    wall_ringing = np.zeros_like(wall_time)
    for amplitude, frequency_mode, decay in zip(
        wall_amplitudes, wall_frequencies, wall_decays
    ):
        wall_ringing += (
            amplitude
            * np.exp(-decay * wall_time)
            * np.sin(2 * np.pi * frequency_mode * wall_time)
        )
    expected_audio[wall_active] = 0.80 * (wall_attack + wall_ringing)
    expected_window = wall_active & (wall_elapsed <= 0.22)
    expected_peak = np.quantile(np.abs(expected_audio[expected_window]), 0.99)
    expected_audio = np.clip(
        0.92 * expected_audio / (expected_peak + 1e-12), -1.0, 1.0
    )

    # One shared 0--2.4 s clock is sufficient here: the observed floor-impact
    # pressure trace is solid ink, while the counterfactual wall-impact
    # template is dashed teal.  Frequency content is intentionally omitted so
    # Figure 1 focuses on event existence and temporal alignment.
    figure = plt.figure(figsize=(12.6, 0.86), dpi=160, facecolor="none")
    waveform_axis = figure.add_axes([0, 0, 1, 1])
    waveform_axis.patch.set_alpha(0)

    waveform_axis.axhline(0, color="#8ea1a8", linewidth=0.35, alpha=0.72)
    waveform_axis.plot(
        time[::8],
        audio[::8],
        color="#344956",
        linewidth=1.15,
        solid_capstyle="round",
        rasterized=True,
    )
    waveform_axis.fill_between(
        time[::8],
        0,
        audio[::8],
        color="#344956",
        alpha=0.075,
        linewidth=0,
    )
    expected_visible = np.where(
        (wall_elapsed >= 0) & (wall_elapsed <= 0.46), expected_audio, np.nan
    )
    waveform_axis.plot(
        time[::8],
        expected_visible[::8],
        color="#23867D",
        linewidth=1.35,
        linestyle=(0, (3.0, 2.0)),
        solid_capstyle="round",
        rasterized=True,
    )
    waveform_axis.set_xlim(0, duration)
    waveform_axis.set_ylim(-1.05, 1.05)
    waveform_axis.axis("off")
    figure.savefig(
        output,
        dpi=160,
        bbox_inches=None,
        pad_inches=0,
        facecolor="none",
        edgecolor="none",
        transparent=True,
    )
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
