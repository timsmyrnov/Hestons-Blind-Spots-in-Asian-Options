"""Full-Heston check: one fixing = European option on a futures, so QuantLib's
semi-analytic Heston price is the exact answer. Skips if QuantLib is absent."""

import pytest
from hestonlab.heston import AsianContract, HestonParameters, MonteCarloConfig, price_asian_options

ql = pytest.importorskip("QuantLib")

F0, T = 80.0, 0.5
PARAMS = HestonParameters(v0=0.09, kappa=2.0, theta=0.06, sigma=0.6, rho=-0.6)


def quantlib_price(strike, side):
    today = ql.Date(1, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    # Zero rate and zero dividend: zero drift, like a futures; discount factor 1.
    flat = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, dc))
    p = PARAMS
    process = ql.HestonProcess(
        flat, flat, ql.QuoteHandle(ql.SimpleQuote(F0)), p.v0, p.kappa, p.theta, p.sigma, p.rho
    )
    option_type = ql.Option.Call if side == "C" else ql.Option.Put
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(option_type, strike),
        ql.EuropeanExercise(today + int(round(T * 365))),
    )
    option.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(process)))
    return option.NPV()


def test_single_fixing_matches_quantlib_heston():
    contracts = [
        AsianContract(f"{s}{k}", (T,), (F0,), 1, 0.0, k, s, 1.0)
        for k in (60, 70, 80, 90, 100)
        for s in ("C", "P")
    ]
    config = MonteCarloConfig(paths=200_000, max_step_years=1 / 365, seed=7)
    for c, e in zip(contracts, price_asian_options(contracts, PARAMS, config)):
        # Errors share one path set, so allow 4 SE plus a small absolute floor.
        assert abs(e.price - quantlib_price(c.strike, c.side)) < 4 * e.standard_error + 0.01
