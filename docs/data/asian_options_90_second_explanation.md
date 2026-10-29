# WTI Asian options — a 90-second explanation

An Asian option depends on an average price over time. For CME's WTI Average Price Option, the relevant average is the arithmetic mean of the nearby CL futures settlement on each applicable business day in the contract month. Its clearing code is AO, its Globex root is AAO, and one contract represents 1,000 barrels. The option is European and cash settled.

For a clearly hypothetical example, suppose the month's daily fixings average $75 per barrel. A $70 call pays $5 per barrel, or $5,000 for one contract. A $70 put pays zero. That calculation is the terminal payoff, before subtracting the option premium.

Averaging makes the contract path dependent. Halfway through the month, some fixings are already known and cannot change. The model must carry their count and sum and price the remaining uncertain fixings. CL also expires during the month, so the correct nearby contract changes. An adjusted continuous oil chart is not the fixing series.

Our archive contains actual exchange AO settlement marks across strikes and maturities. It gives the project a real empirical target. It does not automatically give executable prices or prove the five Heston parameters are identifiable. The model comparison must hold the averaging state and other market inputs consistent and account for exchange valuation conventions.
