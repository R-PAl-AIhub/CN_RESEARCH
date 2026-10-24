"""Unit test: fused closed-form epoch update == autodiff step (hinge / squared hinge / FedProx)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedstat import tfcpu
import numpy as np
import tensorflow as tf
from fedstat.models import LinearSVM, SVMTrainer

rng = np.random.RandomState(0)
d, B = 12, 64
X = rng.randn(B, d).astype(np.float32)
y = (rng.rand(B) > 0.5).astype(np.int8)
w0 = (0.3 * rng.randn(d)).astype(np.float32); b0 = np.float32(0.1)
wg = (0.2 * rng.randn(d)).astype(np.float32); bg = np.float32(-0.05)
worst = 0.0
for sq in (0.0, 1.0):
    for mu in (0.0, 0.7):
        m1 = LinearSVM(d); m2 = LinearSVM(d)
        for m in (m1, m2):
            m.w.assign(w0); m.b.assign(b0)
        t1 = SVMTrainer(m1, batch_size=B); t2 = SVMTrainer(m2, batch_size=B)
        # autodiff reference: a single step on the whole batch
        t1._step(X, (2.0 * y - 1.0).astype(np.float32), np.float32(0.1), np.float32(1e-3), np.float32(mu), np.float32(sq), wg, bg)
        # fused epoch with exactly one full batch (perm = identity)
        t2._epoch_fn(B)(tf.constant(X), tf.constant((2.0 * y - 1.0).astype(np.float32)), tf.constant(np.arange(B, dtype=np.int32)),
                        tf.constant(0.1), tf.constant(1e-3), tf.constant(mu, tf.float32), tf.constant(sq), tf.constant(wg), tf.constant(bg))
        diff = max(np.abs(m1.w.numpy() - m2.w.numpy()).max(), abs(float(m1.b.numpy() - m2.b.numpy())))
        worst = max(worst, diff)
        print(f"sq={sq} mu={mu}: max |autodiff - closed form| = {diff:.2e}")
assert worst < 1e-5, worst
print("PASS")
