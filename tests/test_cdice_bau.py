"""Tests for ``cdice_bau``: the CDICE business-as-usual model of Folini, Friedl,
Kübler & Scheidegger (2024), multi-model-mean calibration.

Guards:
  1. Wiring: dims, names, bounds (softplus on three outputs, linear on four).
  2. Exogenous paths at several years against numbers computed in float64
     from the reference implementation's formulas.
  3. The clock: tau advances exactly one year per step.
  4. GOLDEN against the stored reference solution: from the reference's 2015
     state and policy, our transition reproduces its 2016 state, our
     residuals reproduce its recorded Euler discrepancies, and our social
     cost of carbon reproduces its 2015 value.
  5. Off-equilibrium: transition and residuals equal an independent numpy
     transcription of the reference formulas at random points where every
     residual is far from zero (the on-path rows alone are too close to zero
     to catch a dropped term).
  6. Smoke train in the smoke convention.

Reference rows are copied from the public replication package
github.com/ClimateChangeEcon/Climate_in_Climate_Economics,
DEQN_for_IAMs/gdice_baseline/bau_results/BAU_cdice/ (states.csv, ps.csv,
defs.csv, simulated_euler_discrepancies_2015-2100.csv), first rows.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deqn_jax.models import load_model
from deqn_jax.models.cdice_bau import exogenous as ex

MODEL = load_model("cdice_bau")
C = MODEL.constants


# --------------------------------------------------------------------------- #
# 1. Wiring
# --------------------------------------------------------------------------- #
def test_dims_and_names():
    assert MODEL.n_states == 7 and MODEL.n_policies == 7 and MODEL.n_shocks == 0
    assert MODEL.state_names[-1] == "tau"
    assert len(MODEL.equation_names) == 7
    assert MODEL.steady_state_fn is None and MODEL.init_state_fn is not None


def test_init_state_is_the_2015_state():
    s = MODEL.init_state_fn(jax.random.PRNGKey(0), 3, C)
    assert s.shape == (3, 7)
    np.testing.assert_allclose(
        np.asarray(s[1]), [2.926, 0.851, 0.628, 1.323, 1.1, 0.27, 0.0]
    )


def test_output_bounds_softplus_and_linear():
    from deqn_jax.networks.mlp import create_mlp

    net = create_mlp(
        7,
        7,
        hidden_sizes=(16,),
        policy_lower=MODEL.policy_lower,
        policy_upper=MODEL.policy_upper,
        key=jax.random.PRNGKey(1),
    )
    x = jax.random.normal(jax.random.PRNGKey(2), (256, 7)) * 3.0
    out = net(x)
    assert bool(jnp.all(out[:, :3] > 0.0))
    assert bool(jnp.any(out[:, 3:] < 0.0)) and bool(jnp.any(out[:, 3:] > 0.0))
    raw = jax.vmap(lambda v: net.layers[1](jax.nn.tanh(net.layers[0](v))))(x)
    np.testing.assert_allclose(np.asarray(out[:, 3:]), np.asarray(raw[:, 3:]))
    np.testing.assert_allclose(
        np.asarray(out[:, :3]), np.asarray(jax.nn.softplus(raw[:, :3])), rtol=1e-12
    )
    grads = jax.grad(lambda v: jnp.sum(net(v)))(x[0])
    assert bool(jnp.all(jnp.isfinite(grads)))


# --------------------------------------------------------------------------- #
# 2. Exogenous paths (and 3. the clock)
# --------------------------------------------------------------------------- #
# Float64 evaluation of the reference's DEQN_for_IAMs/gdice_baseline/
# Definitions.py (Tstep = 1, gdice_baseline_mmm_mmm constants), transcribed
# independently of this package. Columns:
# t: (tau, tfp, gr_tfp, lab, gr_lab, sigma, theta1, Eland, Fex, beta_hat)
# The reference's own exoparams.csv (float32) agrees to ~1e-6, except sigma,
# whose float32 (1 + deltaSigma)**t drifts to 7e-5 relative by 2100.
EXPECTED = {
    0.0: (0.0, 0.010295, 0.0217, 7403.0, 0.014831770903687697, 9.55592e-05,
          0.0741061596, 0.00070922, 0.5, 0.9901159423692696),
    1.0: (0.014888060396937353, 0.01052027324449447, 0.021591770798481208,
          7511.34134151095, 0.01423128668867068, 9.41169682672291e-05,
          0.07262368117446268, 0.0006930940987416836, 0.5058823529411764,
          0.9895697658302143),
    10.0: (0.1392920235749422, 0.012721870998175155, 0.020641678511665495,
           8366.172820821566, 0.010038827812994481, 8.20216956467638e-05,
           0.06050563474382231, 0.0005634991215674145, 0.5588235294117647,
           0.9858511220436952),
    85.0: (0.7205690317785927, 0.04626091670709259, 0.014186804337317687,
           11080.102649620803, 0.0010156267812688206, 2.481071406549297e-05,
           0.01257899403032621, 0.00010040035816101373, 1.0,
           0.979837600090969),
    100.0: (0.7768698398515702, 0.056787445838459474, 0.013161715315764143,
            11219.096757429923, 0.0006710172007289664, 1.932092294532687e-05,
            0.00908787677479184, 7.110557714508684e-05, 1.0,
            0.9799519346869215),
    200.0: (0.950212931632136, 0.1599820410978558, 0.007982983873420299,
            11480.740387677231, 4.4958564763315675e-05, 3.3022292681520456e-06,
            0.0009420946609144045, 7.1289629468090316e-06, 1.0,
            0.9816235600242349),
}  # fmt: skip

_PATHS = (
    ex.tfp, ex.gr_tfp, ex.lab, ex.gr_lab, ex.sigma,
    ex.theta1, ex.eland, ex.fex, ex.beta_hat,
)  # fmt: skip


@pytest.mark.parametrize("t", sorted(EXPECTED))
def test_exogenous_paths_match_reference_formulas(t):
    expected = EXPECTED[t]
    tau = jnp.asarray(expected[0])
    t_back = ex.real_time(tau, C)
    assert float(t_back) == pytest.approx(t, abs=1e-9)
    for fn, want in zip(_PATHS, expected[1:]):
        got = float(fn(t_back, C))
        assert got == pytest.approx(want, rel=1e-10, abs=1e-15), fn.__name__


def test_clock_advances_one_year():
    tau = jnp.array([0.0, 0.3, 0.9, 0.999])
    t = ex.real_time(tau, C)
    np.testing.assert_allclose(
        np.asarray(ex.real_time(ex.next_tau(tau, C), C)), np.asarray(t + 1.0), rtol=1e-9
    )


# --------------------------------------------------------------------------- #
# 4. Golden against the stored reference solution (BAU_cdice)
# --------------------------------------------------------------------------- #
# states.csv rows 2015, 2016 (kx in trillion USD, carbon in GtC):
REF_STATES = (
    (223.00183, 851.0, 628.0, 1323.0, 1.1, 0.27, 0.0),
    (228.32054, 857.90106, 630.85046, 1324.0083, 1.155917, 0.27571872, 0.014888048),
)
# ps.csv rows 2015, 2016 (kplusy in trillion USD):
REF_POLICIES = (
    (228.4021, 0.9839649, 1.2317625, -0.92852783, -0.14035206, -0.03941512,
     -0.27845266),
    (234.1357, 0.99036807, 1.2281517, -0.92367494, -0.139298, -0.04043092,
     -0.2790674),
)  # fmt: skip
# simulated_euler_discrepancies_2015-2100.csv, first row (|residual|, 2015):
REF_EULER_2015 = {
    "foc_k": 1.862e-04,
    "budget": 1.402e-03,
    "foc_tat": 4.978e-05,
    "foc_mat": 2.109e-04,
    "foc_muo": 1.432e-04,
    "foc_mlo": 1.341e-07,
    "foc_toc": 4.384e-05,
}
# defs.csv row 2015: social cost of carbon, USD per tCO2.
REF_SCC_2015 = 25.337843


def _normalize(row_s, row_p):
    """Reference post-processed units -> model units (effective labour, 1000 GtC)."""
    t = ex.real_time(jnp.asarray(row_s[6]), C)
    scale = ex.tfp(t, C) * ex.lab(t, C)
    s = jnp.array(
        [row_s[0] / scale, row_s[1] / 1e3, row_s[2] / 1e3, row_s[3] / 1e3, *row_s[4:]]
    )
    p = jnp.array([row_p[0] / (ex.growth_factor(t, C) * scale), *row_p[1:]])
    return s[None, :], p[None, :]


def test_transition_reproduces_reference_2016_state():
    s0, p0 = _normalize(REF_STATES[0], REF_POLICIES[0])
    s1, _ = _normalize(REF_STATES[1], REF_POLICIES[1])
    nxt = MODEL.step_fn(s0, p0, jnp.zeros((1, 0)), C)
    np.testing.assert_allclose(np.asarray(nxt), np.asarray(s1), rtol=2e-6)


def test_reference_solution_has_the_reference_residuals():
    s0, p0 = _normalize(REF_STATES[0], REF_POLICIES[0])
    _, p1 = _normalize(REF_STATES[1], REF_POLICIES[1])
    s1 = MODEL.step_fn(s0, p0, jnp.zeros((1, 0)), C)
    res = MODEL.equations_fn(s0, p0, s1, p1, C)
    assert tuple(res) == MODEL.equation_names
    for name, want in REF_EULER_2015.items():
        got = abs(float(res[name][0]))
        # CSV values carry ~8 significant digits (float32); residuals are
        # differences of O(1) terms, so agreement is to ~1e-6 absolute.
        assert got == pytest.approx(want, abs=2e-6), name


def test_scc_2015_matches_reference():
    s0, p0 = _normalize(REF_STATES[0], REF_POLICIES[0])
    scc = float(MODEL.definitions_fn(s0, p0, C)["scc"][0]) / C["c2co2"]
    assert scc == pytest.approx(REF_SCC_2015, rel=1e-5)


# --------------------------------------------------------------------------- #
# 5. Off-equilibrium golden: an independent numpy transcription of the
#    reference's gdice_baseline Definitions.py / Equations.py (Tstep = 1, its
#    variable names), evaluated at random states and policies where every
#    residual is far from zero, so a dropped or altered term cannot hide
#    under the tolerance of the on-path golden rows above.
# --------------------------------------------------------------------------- #
_P = {
    "vartheta": 0.015, "L0": 7403.0, "Linfty": 11500.0, "deltaL": 0.0268,
    "A0hat": 0.010295, "gA0hat": 0.0217, "deltaA": 0.005,
    "sigma0": 0.0000955592, "gSigma0": -0.0152, "deltaSigma": 0.001,
    "ELand0": 0.00070922, "deltaLand": 0.023, "fex0": 0.5, "fex1": 1.0,
    "rho": 0.015, "psi": 0.68965517, "alpha": 0.3, "delta": 0.1,
    "pi1": 0.0, "pi2": 0.00236, "pow1": 1.0, "pow2": 2.0,
    "b12_": 0.054, "b23_": 0.0082, "MATeq": 0.607, "MUOeq": 0.489,
    "MLOeq": 1.281, "c1_": 0.137, "c3_": 0.73, "c4_": 0.00689,
    "f2xco2": 3.45, "t2xco2": 3.25, "MATbase": 0.607,
}  # fmt: skip


def _np_exo(tau):
    P = _P
    t = -np.log(1 - tau) / P["vartheta"]
    tfp = P["A0hat"] * np.exp(
        P["gA0hat"] * (1 - np.exp(-P["deltaA"] * t)) / P["deltaA"]
    )
    gr_tfp = P["gA0hat"] * np.exp(-P["deltaA"] * t)
    lab = P["L0"] + (P["Linfty"] - P["L0"]) * (1 - np.exp(-P["deltaL"] * t))
    gr_lab = P["deltaL"] / (
        (P["Linfty"] / (P["Linfty"] - P["L0"])) * np.exp(P["deltaL"] * t) - 1
    )
    ds = P["deltaSigma"]
    sigma = P["sigma0"] * np.exp(P["gSigma0"] / np.log(1 + ds) * ((1 + ds) ** t - 1))
    eland = P["ELand0"] * np.exp(-P["deltaLand"] * t)
    fex = P["fex0"] + (1 / 85) * (P["fex1"] - P["fex0"]) * np.minimum(t, 85)
    beta_hat = np.exp(-P["rho"] + (1 - 1 / P["psi"]) * gr_tfp + gr_lab)
    return t, tfp, gr_tfp, lab, gr_lab, sigma, eland, fex, beta_hat


def _np_reference(state, ps, ps_next):
    P = _P
    a, psi, d = P["alpha"], P["psi"], P["delta"]
    b12, b23 = P["b12_"], P["b23_"]
    b21, b32 = P["MATeq"] / P["MUOeq"] * b12, P["MUOeq"] / P["MLOeq"] * b23
    c1, c1c3, c4 = P["c1_"], P["c1_"] * P["c3_"], P["c4_"]
    c1f = P["c1_"] * P["f2xco2"] / P["t2xco2"]

    def omega(T):
        return P["pi1"] * T ** P["pow1"] + P["pi2"] * T ** P["pow2"]

    def omega_prime(T):
        return P["pow1"] * P["pi1"] * T ** (P["pow1"] - 1) + P["pow2"] * P[
            "pi2"
        ] * T ** (P["pow2"] - 1)

    kx, MAT, MUO, MLO, TAT, TOC, tau = state.T
    t, tfp, gr_tfp, lab, gr_lab, sigma, eland, fex, bh = _np_exo(tau)
    kplus, lam, nuAT, nuUO, nuLO, etaAT, etaOC = ps.T
    MATp = (1 - b12) * MAT + b21 * MUO + sigma * tfp * lab * kx**a + eland
    MUOp = b12 * MAT + (1 - b21 - b23) * MUO + b32 * MLO
    MLOp = b23 * MUO + (1 - b32) * MLO
    TATp = (
        (1 - c1c3 - c1f) * TAT
        + c1c3 * TOC
        + c1 * (P["f2xco2"] * (np.log(MAT / P["MATbase"]) / np.log(2.0)) + fex)
    )
    TOCp = c4 * TAT + (1 - c4) * TOC
    taup = 1 - np.exp(-P["vartheta"] * (t + 1))
    nxt = np.stack([kplus, MATp, MUOp, MLOp, TATp, TOCp, taup], axis=1)
    _, tfp_n, _, lab_n, _, sigma_n, _, _, _ = _np_exo(taup)
    lam_n, nuAT_n, nuUO_n, nuLO_n, etaAT_n, etaOC_n = ps_next.T[1:]
    growth = np.exp(gr_tfp + gr_lab)
    res = {
        "foc_k": growth * lam - bh * (
            lam_n * ((1 - omega(TATp)) * a * kplus ** (a - 1) + (1 - d))
            + (-nuAT_n) * sigma_n * tfp_n * lab_n * a * kplus ** (a - 1)
        ),
        "budget": (1 - omega(TAT)) * kx**a - lam ** (-psi) + (1 - d) * kx
        - growth * kplus,
        "foc_tat": etaAT - bh * (
            lam_n * (-omega_prime(TATp)) * kplus**a
            + etaAT_n * (1 - c1c3 - c1f) + etaOC_n * c4
        ),
        "foc_mat": (-nuAT) - bh * (
            (-nuAT_n) * (1 - b12) + nuUO_n * b12
            + etaAT_n * c1 * P["f2xco2"] * (1 / (np.log(2.0) * MATp))
        ),
        "foc_muo": nuUO - bh * (
            (-nuAT_n) * b21 + nuUO_n * (1 - b21 - b23) + nuLO_n * b23
        ),
        "foc_mlo": nuLO - bh * (nuUO_n * b32 + nuLO_n * (1 - b32)),
        "foc_toc": etaOC - bh * (etaAT_n * c1c3 + etaOC_n * (1 - c4)),
    }  # fmt: skip
    return nxt, res


def _random_points(seed, n=32):
    rng = np.random.default_rng(seed)
    s_lo = [2.0, 0.8, 0.6, 1.3, 0.5, 0.2, 0.0]
    s_hi = [4.0, 2.0, 1.2, 1.6, 5.0, 1.5, 0.99]
    p_lo = [2.0, 0.6, 0.5, -1.8, -0.4, -0.3, -0.9]
    p_hi = [4.0, 1.4, 2.0, -0.4, 0.1, 0.1, -0.1]
    state = rng.uniform(s_lo, s_hi, size=(n, 7))
    return state, rng.uniform(p_lo, p_hi, (n, 7)), rng.uniform(p_lo, p_hi, (n, 7))


@pytest.mark.parametrize("seed", [0, 1])
def test_off_equilibrium_matches_reference_transcription(seed):
    state, ps, ps_next = _random_points(seed)
    want_next, want = _np_reference(state, ps, ps_next)
    s, p, pn = jnp.asarray(state), jnp.asarray(ps), jnp.asarray(ps_next)
    got_next = MODEL.step_fn(s, p, jnp.zeros((len(state), 0)), C)
    np.testing.assert_allclose(np.asarray(got_next), want_next, rtol=1e-12)
    got = MODEL.equations_fn(s, p, got_next, pn, C)
    for name in MODEL.equation_names:
        # Every residual is far from zero here, so a missing term shows.
        assert np.min(np.abs(want[name])) > 1e-4, name
        np.testing.assert_allclose(
            np.asarray(got[name]), want[name], rtol=1e-10, atol=1e-13, err_msg=name
        )


# --------------------------------------------------------------------------- #
# 6. Smoke train (3 episodes, hidden=(16,), batch=16)
# --------------------------------------------------------------------------- #
def test_smoke_train():
    from deqn_jax.config import NetworkConfig, OptimizerConfig, TrainConfig
    from deqn_jax.training.trainer import train_from_config

    cfg = TrainConfig(
        model="cdice_bau",
        episodes=3,
        batch_size=16,
        sim_batch=16,
        episode_length=20,
        mc_samples=1,
        initialize_each_episode=True,
        network=NetworkConfig(hidden_sizes=(16,), activation="relu"),
        optimizer=OptimizerConfig(learning_rate=1e-4),
        verbose=False,
        seed=0,
    )
    _state, history = train_from_config(cfg)
    losses = history["loss"]
    assert len(losses) == 3
    assert all(math.isfinite(float(v)) for v in losses)
