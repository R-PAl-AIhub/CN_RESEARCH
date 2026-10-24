"""
TensorFlow/Keras models.

LinearSVM : f(x) = w.x + b trained with the L2-regularised hinge loss (Eq. 4 of the senior paper, mean form).
            This is the TF port of the senior LinearSVC detector and the model used for federated averaging.
MLP       : 17->100->50->1 ReLU network (senior ANN baseline, ported from sklearn MLPClassifier to Keras).
"""
from . import tfcpu  # noqa: F401  (forces CPU before TF is used)
import numpy as np
import tensorflow as tf


class LinearSVM(tf.keras.Model):
    """f(x) = x @ w + b.  Labels are mapped to {-1,+1} inside the loss."""

    def __init__(self, d: int, name="linear_svm"):
        super().__init__(name=name)
        self.w = tf.Variable(tf.zeros((d,), tf.float32), name="w")
        self.b = tf.Variable(0.0, name="b")

    def call(self, x):
        return tf.linalg.matvec(x, self.w) + self.b

    # ---- parameter (de)serialisation used by the federated server ------------------------------
    def get_params(self) -> np.ndarray:
        return np.concatenate([self.w.numpy(), [self.b.numpy()]]).astype(np.float64)

    def set_params(self, p: np.ndarray) -> None:
        self.w.assign(p[:-1].astype(np.float32))
        self.b.assign(np.float32(p[-1]))

    def n_params(self) -> int:
        return int(self.w.shape[0]) + 1


def _make_hinge_step(model: LinearSVM):
    @tf.function(reduce_retracing=True)
    def step(x, y_pm, lr, lam, mu, sq, wg, bg):
        # loss = mean[hinge or squared hinge] + lam/2 ||w||^2 + mu/2 ||theta - theta_global||^2
        # sq=0 -> hinge (paper, Eq. 4);  sq=1 -> squared hinge (sklearn LinearSVC default, what the senior code runs)
        # mu>0 turns the local objective into the FedProx objective.
        with tf.GradientTape() as t:
            f = model(x)
            margin = tf.nn.relu(1.0 - y_pm * f)
            data = (1.0 - sq) * tf.reduce_mean(margin) + sq * tf.reduce_mean(tf.square(margin))
            loss = (data + 0.5 * lam * tf.reduce_sum(tf.square(model.w))
                    + 0.5 * mu * (tf.reduce_sum(tf.square(model.w - wg)) + tf.square(model.b - bg)))
        g = t.gradient(loss, [model.w, model.b])
        model.w.assign_sub(lr * g[0])
        model.b.assign_sub(lr * g[1])
        return loss
    return step


def _make_epoch_fn(model: LinearSVM, batch_size: int):
    """One full pass over a client's data as a SINGLE tf.function (closed-form mini-batch gradients).
    Gradient of  mean l(y f) + lam/2 ||w||^2 + mu/2 ||theta - theta_g||^2  with f = w.x + b:
        dl/df = -y 1[m>0]  (hinge)   or   -2 m y  (squared hinge),  m = max(0, 1 - y f)
        g_w = X_b^T (dl/df)/|b| + lam w + mu (w - w_g),   g_b = sum(dl/df)/|b| + mu (b - b_g)
    (numerically equal to the autodiff step above; see code/tests/test_gradients.py)."""
    B = batch_size

    @tf.function(reduce_retracing=True)
    def epoch(X, y_pm, perm, lr, lam, mu, sq, wg, bg):
        n = tf.shape(X)[0]
        nsteps = (n + B - 1) // B
        for i in tf.range(nsteps):
            j = perm[i * B:(i + 1) * B]
            xb = tf.gather(X, j)
            yb = tf.gather(y_pm, j)
            f = tf.linalg.matvec(xb, model.w) + model.b
            m = tf.nn.relu(1.0 - yb * f)
            coef = (1.0 - sq) * (-yb * tf.cast(m > 0.0, tf.float32)) + sq * (-2.0 * m * yb)
            coef = coef / tf.cast(tf.shape(xb)[0], tf.float32)
            gw = tf.linalg.matvec(xb, coef, transpose_a=True) + lam * model.w + mu * (model.w - wg)
            gb = tf.reduce_sum(coef) + mu * (model.b - bg)
            model.w.assign_sub(lr * gw)
            model.b.assign_sub(lr * gb)
    return epoch


class SVMTrainer:
    """Mini-batch SGD trainer for LinearSVM (plain SGD - the local solver used in FedAvg/FedProx)."""

    def __init__(self, model: LinearSVM, batch_size: int = 256):
        self.model = model
        self._step = _make_hinge_step(model)          # autodiff reference step (used by the unit test)
        self._epochs = {}
        self.batch_size = batch_size

    def _epoch_fn(self, bs):
        if bs not in self._epochs:
            self._epochs[bs] = _make_epoch_fn(self.model, bs)
        return self._epochs[bs]

    def loss(self, X, y, lam, squared, mu=0.0, global_params=None):
        pm = tf.constant(2.0 * y - 1.0, tf.float32)
        m = tf.nn.relu(1.0 - pm * self.model(tf.constant(X, tf.float32)))
        data = tf.reduce_mean(tf.square(m) if squared else m)
        return float(data + 0.5 * lam * tf.reduce_sum(tf.square(self.model.w)))

    def fit(self, X, y, epochs=1, batch_size=None, lr=0.05, lam=1e-4, mu=0.0, rng=None, global_params=None,
            squared=False, lr_end=None):
        """lr_end: if given, the step size decays linearly (per epoch) from lr to lr_end over the call."""
        bs = batch_size or self.batch_size
        rng = rng or np.random.RandomState(0)
        d = X.shape[1]
        wg = tf.constant(global_params[:-1] if global_params is not None else np.zeros(d), tf.float32)
        bg = tf.constant(global_params[-1] if global_params is not None else 0.0, tf.float32)
        Xt = tf.constant(X, tf.float32)
        yt = tf.constant((2.0 * y - 1.0), tf.float32)
        ep = self._epoch_fn(bs)
        for e in range(epochs):
            cur = lr if lr_end is None else lr + (lr_end - lr) * (e / max(epochs - 1, 1))
            perm = tf.constant(rng.permutation(len(X)).astype(np.int32))
            ep(Xt, yt, perm, tf.constant(cur, tf.float32), tf.constant(lam, tf.float32), tf.constant(mu, tf.float32),
               tf.constant(1.0 if squared else 0.0, tf.float32), wg, bg)
        return self.loss(X, y, lam, squared)

    def predict(self, X):
        return (self.model(tf.constant(X, tf.float32)).numpy() > 0).astype(np.int8)

    def decision(self, X):
        return self.model(tf.constant(X, tf.float32)).numpy()


def make_mlp(d: int, hidden=(100, 50), lr=1e-3, l2=1e-3) -> tf.keras.Model:
    """Keras port of sklearn MLPClassifier(hidden=(100,50), relu, adam, alpha=1e-3)."""
    reg = tf.keras.regularizers.l2(l2 / 2.0)
    layers = [tf.keras.layers.Input((d,))]
    layers += [tf.keras.layers.Dense(h, activation="relu", kernel_regularizer=reg) for h in hidden]
    layers += [tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=reg)]
    m = tf.keras.Sequential(layers)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return m
