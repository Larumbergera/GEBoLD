from itertools import pairwise

import numpy as np
import pytest

from gebold.gaze import Adam, FineTuning, GazeModel, elu, elu_grad, pog_loss

LAYERS = [16, 32, 16, 16, 8, 8, 2]  # DEFC


def random_model(seed=0):
    rng = np.random.default_rng(seed)
    weights = []
    for n_in, n_out in pairwise(LAYERS):
        weights += [rng.normal(0, 1 / np.sqrt(n_in), (n_in, n_out)), rng.normal(0, 0.1, n_out)]
    return GazeModel(weights, {"license": "test"})


def data(n=12, seed=1):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, (n, 16)), rng.uniform(-1, 1, (n, 2))


def test_elu_and_gradient():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 3.0])
    np.testing.assert_allclose(elu(x), [np.expm1(-2), np.expm1(-0.5), 0, 0.5, 3])
    eps = 1e-6
    np.testing.assert_allclose(
        elu_grad(x[[0, 1, 3, 4]]),
        (elu(x + eps) - elu(x - eps))[[0, 1, 3, 4]] / (2 * eps),
        rtol=1e-6,
    )


def test_pog_loss_is_mean_euclidean_distance():
    loss, grad = pog_loss(np.array([[3.0, 4.0], [0.0, 0.0]]), np.zeros((2, 2)))
    assert loss == 2.5
    np.testing.assert_allclose(grad, [[0.3, 0.4], [0.0, 0.0]])


def test_backpropagation_matches_finite_differences():
    model, (x, y) = random_model(), data(4)

    def loss_of(weights):
        m = GazeModel(weights)
        return pog_loss(m.predict(x), y)[0]

    class Capture(Adam):
        def update(self, params, grads):
            self.grads = [g.copy() for g in grads]

    opt = Capture(lr=0.0)
    model.train_step(x, y, opt)
    eps = 1e-6
    for k in (0, 1, 6, 11):  # first and last layers, kernels and biases
        w = model.weights[k]
        idx = tuple(0 for _ in w.shape)
        plus = [v.copy() for v in model.weights]
        minus = [v.copy() for v in model.weights]
        plus[k][idx] += eps
        minus[k][idx] -= eps
        numeric = (loss_of(plus) - loss_of(minus)) / (2 * eps)
        assert opt.grads[k][idx] == pytest.approx(numeric, rel=1e-5, abs=1e-9)


def test_adam_first_step_moves_by_lr():
    p, g = [np.array([1.0, -1.0])], [np.array([0.5, -2.0])]
    Adam(lr=0.1).update(p, g)
    np.testing.assert_allclose(p[0], [0.9, -0.9], rtol=1e-6)


def test_save_and_load_keep_weights_and_metadata(tmp_path):
    model = random_model()
    model.save(tmp_path / "m.npz")
    loaded = GazeModel.load(tmp_path / "m.npz")
    assert loaded.metadata == {"license": "test"}
    x, _ = data()
    np.testing.assert_allclose(loaded.predict(x), model.predict(x))


def test_predict_single_and_batch():
    model, (x, _) = random_model(), data(3)
    assert model.predict(x[0]).shape == (2,)
    np.testing.assert_allclose(model.predict(x)[0], model.predict(x[0]))
    with pytest.raises(ValueError, match="16 features"):
        model.predict(np.zeros(15))


def test_fine_tune_reduces_loss_and_leaves_copy_untouched():
    base, (x, y) = random_model(), data()
    calibrated = base.copy()
    history = calibrated.fine_tune(x, y, FineTuning(epochs=50, lr=1e-3))
    assert history[-1] < history[0]
    assert not np.allclose(calibrated.weights[0], base.weights[0])
    np.testing.assert_allclose(base.weights[0], random_model().weights[0])


def test_fine_tune_rejects_mismatched_samples():
    with pytest.raises(ValueError, match="same"):
        random_model().fine_tune(np.zeros((3, 16)), np.zeros((2, 2)))


@pytest.mark.tf
def test_fine_tune_matches_keras():
    tf = pytest.importorskip("tensorflow")
    base, (x, y) = random_model(), data(6)
    inp = tf.keras.Input((16,))
    h = inp
    for i, units in enumerate(LAYERS[1:]):
        h = tf.keras.layers.Dense(units, activation="linear" if i == 5 else "elu")(h)
    keras_model = tf.keras.Model(inp, h)
    keras_model.set_weights(base.weights)

    def pog_euc(t, p):  # losses.pog_euc of the thesis code
        return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(t - p), axis=1)))

    keras_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=pog_euc)
    keras_model.fit(x, y, batch_size=2, epochs=5, shuffle=False, verbose=0)
    ours = base.copy()
    ours.fine_tune(x, y, FineTuning(epochs=5, batch_size=2, lr=1e-3, shuffle=False))
    for a, b in zip(keras_model.get_weights(), ours.weights, strict=True):
        np.testing.assert_allclose(a, b, atol=2e-6)
