"""Independent numerical invariants of the native Monte Carlo engine."""

import numpy as np
import pytest
from scipy.special import ndtr
from hestonlab.cpp_pricer import CppPricer
from hestonlab.heston import AsianContract, HestonParameters, MonteCarloConfig, price_asian_options


def contract(name="a", side="C", strike=80, forward=80, times=(0.1, 0.2, 0.3)):
    return AsianContract(name, times, (forward,) * len(times), len(times), 0, strike, side, 0.98)


def test_fully_accrued_payoffs_are_exact_and_have_zero_covariance():
    contracts = [
        AsianContract("c", (), (), 20, 1700, 80, "C", 0.97),
        AsianContract("p", (), (), 20, 1700, 90, "P", 0.97),
    ]
    got = CppPricer(contracts).price(HestonParameters(), paths=32, covariance=True)
    np.testing.assert_allclose(got.prices, [4.85, 4.85], atol=1e-12)
    assert not got.standard_errors.any() and not got.covariance.any()


def test_constant_variance_terminal_price_matches_black_limit():
    c = contract(times=(0.5,))
    got = CppPricer([c]).price([0.04, 2, 0.04, 1e-10, -0.5], paths=65536)
    s = 0.2 * np.sqrt(0.5)
    exact = 0.98 * 80 * (ndtr(s / 2) - ndtr(-s / 2))
    assert abs(got.prices[0] - exact) < 5 * got.standard_errors[0] + 0.003


def test_scaled_contracts_share_paths_and_covariance():
    a = contract()
    b = contract("b", forward=160, strike=160)
    got = CppPricer([a, b]).price(HestonParameters(), paths=4096, covariance=True)
    assert got.prices[1] == pytest.approx(2 * got.prices[0], abs=1e-12)
    assert got.standard_errors[1] == pytest.approx(2 * got.standard_errors[0], abs=1e-12)
    assert got.covariance[0, 1] == pytest.approx(2 * got.covariance[0, 0], abs=1e-12)
    assert np.linalg.eigvalsh(got.covariance).min() > -1e-12


def test_native_and_independent_python_implementation_agree():
    contracts = [
        contract(),
        contract("put", "P", 85),
        contract("long", strike=90, times=(0.2, 0.4, 0.6)),
    ]
    params = HestonParameters(0.08, 1.3, 0.05, 0.7, -0.65)
    got = CppPricer(contracts).price(params, paths=65536, seed=17)
    ref = price_asian_options(contracts, params, MonteCarloConfig(65536, 1 / 365, 891))
    mean = np.array([x.price for x in ref])
    se = np.hypot(got.standard_errors, [x.standard_error for x in ref])
    assert np.all(np.abs(got.prices - mean) < 5 * se + 0.005)


def test_common_random_numbers_are_reproducible_and_reject_bad_inputs():
    pricer = CppPricer([contract()])
    a = pricer.price(HestonParameters(), paths=128, seed=41)
    b = pricer.price(HestonParameters(), paths=128, seed=41)
    np.testing.assert_array_equal(a.prices, b.prices)
    with pytest.raises(ValueError):
        pricer.price([0.04, 2, 0.04, 0.5, 1.1])
    with pytest.raises(ValueError):
        pricer.price(HestonParameters(), paths=129)
    with pytest.raises(ValueError):
        pricer.price(HestonParameters(), max_step=0)
