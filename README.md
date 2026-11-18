# Heston's Blind Spots in Asian Options

We study where Heston's stochastic-volatility model fails to match observed prices of CME WTI Average Price Options.

The study covers 2022–2025, using weekly snapshots on a common strike–maturity grid. C++ Monte Carlo generates Heston's training prices. Separate neural networks embed the Heston and market sets as five-dimensional manifolds in the full 21-price space. Each market network uses the preceding 120 trading days. We measure local projection gaps in that space and check Heston's neural prices against independent C++ simulations.

The licensed settlement data stays outside Git. Missing prices are not imputed.

![Projected slices of Heston's manifold in purple and the market manifold in blue, with green tangent planes](docs/media/manifold-slice.png)

Projected local slices: Heston in purple, the market model in blue. Lines show price differences; green patches show local tangent planes on Heston's slice. The blue surface represents prices modeled from market data, not the quotes themselves.

[Neural methodology and running the code](docs/neural_manifolds.md) · [Earlier C++ workflow](docs/weekly_cpp_runbook.md)
