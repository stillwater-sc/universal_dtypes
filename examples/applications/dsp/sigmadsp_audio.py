"""ADI SigmaDSP-style audio pipeline in q5_23 (5.23, 28-bit).

Analog Devices SigmaDSP audio processors (ADAU series) work in a 5.23 fixed-point
format — 23 fractional bits for audio-grade fidelity, and 5 integer bits (range
±16) for the *headroom* a mixing console needs when many channels sum above unity.
This shows both properties against a float64 reference.

Part A — fidelity. Apply a gain to a full-scale-ish sine and measure the
signal-to-quantization-noise ratio:

    q5_23 : ~136 dB   (well beyond 24-bit audio; the point of 5.23)
    q15   :  ~87 dB   (16-bit: fine for playback, not for a processing chain)
    fp16  :  ~69 dB

Part B — headroom. Sum five 0.5-amplitude channels (peak ≈ 2.0). q5_23's ±16
range passes it untouched; q15's ±1 range clips it to 1.0 — audible distortion.

    python examples/applications/dsp/sigmadsp_audio.py
"""

import numpy as np

import universal_dtypes as ud

_n = np.arange(8000)


def _snr_db(out, ref):
    noise = out - ref
    return 10.0 * np.log10(np.sum(ref**2) / max(np.sum(noise**2), 1e-30))


def gain_stage(signal, dtype, g=0.8):
    return (np.array(signal, dtype=dtype) * np.array([g] * signal.size, dtype=dtype)).astype(
        np.float64
    )


def mix(channels, dtype):
    acc = np.zeros(channels[0].size, dtype=dtype)
    for c in channels:
        acc = acc + np.array(c, dtype=dtype)
    return acc.astype(np.float64)


def main():
    # Part A: fidelity
    sig = 0.5 * np.sin(2 * np.pi * 0.013 * _n)
    ref = gain_stage(sig, np.float64)
    print("Part A — gain-stage fidelity (SNR vs float64):")
    snr = {}
    for name, dt in [("q5_23", ud.q5_23), ("q15", ud.q15), ("fp16", ud.fp16)]:
        snr[name] = _snr_db(gain_stage(sig, dt), ref)
        print(f"  {name:6} {snr[name]:7.1f} dB")

    # Part B: headroom
    chans = [0.5 * np.sin(2 * np.pi * f * _n) for f in (0.01, 0.02, 0.03, 0.04, 0.05)]
    ref_peak = float(np.max(np.abs(mix(chans, np.float64))))
    peak = {
        name: float(np.max(np.abs(mix(chans, dt))))
        for name, dt in [("q5_23", ud.q5_23), ("q15", ud.q15)]
    }
    print(f"\nPart B — mix of 5 channels (float64 peak = {ref_peak:.3f}):")
    for name in ("q5_23", "q15"):
        clipped = peak[name] < ref_peak - 0.01
        print(f"  {name:6} peak = {peak[name]:.3f}   {'CLIPPED' if clipped else 'ok'}")

    # fidelity: q5_23 is audio-grade and beats the 16-bit formats
    assert snr["q5_23"] > 120.0
    assert snr["q5_23"] > snr["q15"] > snr["fp16"]
    # headroom: q5_23 (+-16) passes the >1 mix; q15 (+-1) clips to 1.0
    assert abs(peak["q5_23"] - ref_peak) < 0.01
    assert peak["q15"] <= 1.001
    print("\nOK: q5_23 gives audio-grade fidelity AND the mixing headroom q15 lacks.")


if __name__ == "__main__":
    main()
