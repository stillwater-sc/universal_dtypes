"""FIR coefficient quantization in the DSP Q-formats (q7 / q15 / q31).

"How many bits do my filter taps need?" is the first question when you move an
FIR filter onto a fixed-point DSP. This designs a 31-tap low-pass filter in
float64, quantizes the coefficients to each of the standard DSP formats, and
measures what happens to the frequency response — especially the stopband
attenuation, which is what coefficient error erodes first.

The result is the classic fixed-point-DSP lesson:

    ideal (float64) stopband : -58 dB
    q31  (Q1.31, 32-bit)     : -58 dB   — indistinguishable from ideal
    q15  (Q1.15, 16-bit)     : -58 dB   — design preserved at half the storage
    q7   (Q1.7,  8-bit)      : -32 dB   — 8-bit taps wreck the stopband

That q15 keeps the filter is exactly why 16-bit fixed-point DSPs (TI C5000, ADI
ADSP-21xx/Blackfin) have carried real audio/comms filters for decades; q7 does
not have the coefficient precision.

Only the coefficients are quantized here (the response is evaluated in float64) so
the effect measured is purely coefficient quantization.

    python examples/applications/dsp/fir_coefficient_quantization.py
"""

import numpy as np

import universal_dtypes as ud

N = 31  # taps
FC = 0.2  # cutoff (cycles/sample)

# windowed-sinc low-pass, unity DC gain — taps in (-1, 1), so they fit the Q1.x formats
_n = np.arange(N) - (N - 1) / 2
H = np.sinc(2 * FC * _n) * np.hamming(N)
H = H / H.sum()


def response(taps):
    """Magnitude response |H(f)| on a dense grid over [0, 0.5]."""
    freqs = np.linspace(0.0, 0.5, 257)
    k = np.arange(N)
    return freqs, np.abs((taps[None, :] * np.exp(-2j * np.pi * np.outer(freqs, k))).sum(axis=1))


def stopband_db(freqs, mag):
    return 20 * np.log10(max(mag[freqs > 0.30].max(), 1e-12))


def main():
    freqs, ref = response(H)
    ref_sb = stopband_db(freqs, ref)
    print(f"31-tap low-pass FIR; ideal stopband attenuation = {ref_sb:.1f} dB\n")
    print(f"{'taps':6} {'stopband':>10} {'max |H - Href|':>16}")

    err, sb = {}, {}
    for name, dt in [("q7", ud.q7), ("q15", ud.q15), ("q31", ud.q31)]:
        q = np.array(H, dtype=dt).astype(np.float64)  # quantize the coefficients
        _, mag = response(q)
        err[name] = float(np.max(np.abs(mag - ref)))
        sb[name] = stopband_db(freqs, mag)
        print(f"{name:6} {sb[name]:9.1f}dB {err[name]:16.2e}")

    # 32-bit taps are effectively ideal
    assert err["q31"] < 1e-6
    # 16-bit taps preserve the design (stopband within ~1 dB of ideal)
    assert sb["q15"] <= ref_sb + 1.0 and err["q15"] < 1e-3
    # 8-bit taps badly degrade the stopband (lose >15 dB of attenuation)
    assert sb["q7"] > ref_sb + 15.0
    # finer format -> smaller response error
    assert err["q31"] < err["q15"] < err["q7"]
    print("\nOK: q15/q31 preserve the filter; q7 lacks the coefficient precision.")


if __name__ == "__main__":
    main()
