"""Quantized inference: running an MLP in low precision to save memory.

Inference-time quantization stores weights and activations in a narrow format so
a model needs less memory and bandwidth. The question is always: how much
accuracy do you give up? This runs the *same* fixed 2-layer MLP forward pass in
several number systems and compares each against a float32 reference on two
metrics:

  * argmax agreement — does the quantized net predict the same class?
  * mean relative error of the output logits.

The pattern that falls out:

  * 16-bit types (posit16, fp16, bfloat16) reproduce float32's decisions almost
    exactly at HALF the memory — quantized inference is essentially free here.
  * 8-bit types (fp8e5m2, posit8) QUARTER the memory but visibly degrade — and
    posit8's tapered precision beats fp8e5m2 on this workload.

(NumPy has no BLAS matmul for custom dtypes, so the matmul is done with the
registered elementwise multiply + a sum reduction — which is exactly the point:
these dtypes drop into ordinary NumPy code.)

    python examples/applications/ml/quantized_mlp.py
"""

import numpy as np

import universal_dtypes as ud

rng = np.random.default_rng(0)
N, D_IN, D_HIDDEN, D_OUT = 256, 16, 32, 4

X = rng.standard_normal((N, D_IN)) * 0.5
W1 = rng.standard_normal((D_IN, D_HIDDEN)) / np.sqrt(D_IN)
B1 = rng.standard_normal(D_HIDDEN) * 0.1
W2 = rng.standard_normal((D_HIDDEN, D_OUT)) / np.sqrt(D_HIDDEN)
B2 = rng.standard_normal(D_OUT) * 0.1


def matmul(a, b):
    # (m,k) @ (k,n) via registered elementwise multiply + sum reduction
    return (a[:, :, None] * b[None, :, :]).sum(axis=1)


def forward(dtype):
    x = X.astype(dtype)
    w1, b1 = W1.astype(dtype), B1.astype(dtype)
    w2, b2 = W2.astype(dtype), B2.astype(dtype)
    zero = np.array([0.0], dtype=dtype)
    h = matmul(x, w1) + b1[None, :]
    h = h * (h > zero).astype(dtype)  # ReLU
    out = matmul(h, w2) + b2[None, :]
    return out.astype(np.float64)


def main():
    ref = forward(np.float32)
    ref_pred = np.argmax(ref, axis=1)

    formats = [
        ("float32", np.float32, 4),
        ("posit16", ud.posit16, 2),
        ("fp16", ud.fp16, 2),
        ("bfloat16", ud.bfloat16, 2),
        ("fp8e5m2", ud.fp8e5m2, 1),
        ("posit8", ud.posit8, 1),
    ]
    print(f"{'format':10} {'bytes':>5} {'argmax agree':>13} {'mean rel err':>13}")
    agree, relerr = {}, {}
    for name, dt, nbytes in formats:
        out = forward(dt)
        agree[name] = 100.0 * np.mean(np.argmax(out, axis=1) == ref_pred)
        relerr[name] = float(np.mean(np.abs(out - ref) / (np.abs(ref) + 1e-9)))
        print(f"{name:10} {nbytes:>5} {agree[name]:12.1f}% {relerr[name]:13.3e}")

    print("\n16-bit: ~free — half the memory, float32-level decisions.")
    print("8-bit : 4x smaller, real accuracy cost; posit8 > fp8e5m2 here.")

    # 16-bit types reproduce float32's decisions (viable inference at half memory)
    for name in ("posit16", "fp16", "bfloat16"):
        assert agree[name] >= 99.0, name
        assert relerr[name] < 0.05, name
    # 8-bit types visibly degrade (a real cost) but stay well above chance (25%)
    for name in ("fp8e5m2", "posit8"):
        assert agree[name] < 99.0 and agree[name] > 50.0, name
    # on this workload posit8's tapered precision beats fp8e5m2
    assert agree["posit8"] > agree["fp8e5m2"]
    print("\nOK: 16-bit quantized inference is ~lossless; 8-bit trades accuracy for size.")


if __name__ == "__main__":
    main()
