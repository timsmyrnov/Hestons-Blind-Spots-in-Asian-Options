# Heston-Blind-Spots-in-Asian-Options

Heston has five parameters, so for a fixed set of strikes and maturities, the Asian option prices it can produce form an at most 5-dim set in price space, with one coordinate per option. We use a neural network to fit a manifold to this set. We then measure how real prices deviate from it, and compare it with a manifold learned from real data. Each day's error is the distance between the real prices and the closest point on the fitted manifold.
