# June 2024 AO input validation

Scope: 613 June valuation observations across 19 dates, with August, September and October 2024 averaging months. The retained baseline fits one shared vector using 4,096 paths, 100 evaluations, seed 2024 and a one-day maximum step. All price and RMSE figures below are USD/barrel.

## Fixings and underlying references

**VERIFIED — payoff, contract identity and expiry.** NYMEX Rule 341 defines a cash-settled European option on the average of daily first-nearby CL settlements in the contract month. Premiums are quoted per barrel and the contract multiplier is 1,000. The exchange-sourced definitions in the fitted grid give August 30, September 30 and October 31 as expirations. Each matches the last fixing and the last business day of its month. [CME Rule 341](https://www.cmegroup.com/content/dam/cmegroup/rulebook/NYMEX/3/341.pdf)

**VALID WITH DOCUMENTED ASSUMPTION — fixing dates and roll.** The reconstructed calendar contains 22, 20 and 23 dates respectively. CME's 2024 advisories exclude September 2 from settlements and explain the June 19 gap in valuation dates. The expiring CL contract is included on its final day, with the next contract used on the next settlement day. The three roll boundaries checked were August 20 (CLU4), September 20 (CLV4) and October 22 (CLX4). CME documents a final settlement for expiring CL on that day. Chapter 510 defines the analogous CL average for CSX. CSX provides a monthly cross-check. The pricer uses date- and maturity-specific CL marks for the simulated fixings. [Labor Day](https://www.cmegroup.com/content/dam/cmegroup/tools-information/holiday-calendar/files/labor-day-holiday-settlement-times-2024.pdf) · [Juneteenth](https://www.cmegroup.com/tools-information/holiday-calendar/files/juneteenth-day-settlement-times-2024.pdf) · [CL final settlement](https://www.cmegroup.com/trading/energy/files/NYMEX_Energy_Futures_Final_Settlement_Procedure.pdf) · [CME Rule 510](https://www.cmegroup.com/rulebook/NYMEX/5/510.pdf)

The audit traced 144 distinct archived CME market-data statistics messages delivered by Databento to the normalized CL/CSX records. It checked instrument, date, price, flags and publication identifier. All 1,235 June forward rows match positive final CL records for their valuation date. There are zero duplicate fixing keys. All three averaging months start after June, so every selected option has zero accrued fixings and every realized future fixing value is absent from the pricing inputs. The retrospective calendar and settlement screen have not been certified for a publication-time backtest.

| Averaging month | Fixings | Absolute CL–CSX gap | Gap if rolled a fixing early |
|---|---:|---:|---:|
| August | 22 | 0.00364 | 0.03591 |
| September | 20 | 0.00250 | 0.04350 |
| October | 23 | 0.00174 | 0.01348 |

**UNRESOLVED — terminal feed flags.** One expiring CL fixing in each month and all three terminal CSX marks lack a final flag in the selected feed. The older final-only fixing table leaves those three CL prices blank. The baseline uses the clock-screened table for its calendar and the flagged-final June CL forward marks for pricing. Its reconstructed terminal averages agree with CSX within half a cent per barrel. This supports the pilot mapping. Definitive terminal-price certification would require official historical terminal records. Ben owns that follow-up if needed for a broader seasoned-option study. The June baseline requires no input correction on this evidence.

## Discount and cash timing

**VERIFIED — rate provenance, units and arithmetic.** The 19 selected rates match the archived New York Fed API response and its receipt hash. Rates span 5.31%–5.35%. The availability proxy selects the next Fed publication business day, consistent with the NY Fed's next-business-day morning publication for the previous value date. Every assumed availability date is on or before the June valuation date. Original intraday rate vintages were unavailable. The interval is 63–150 calendar days from valuation to expiry. The pricer receives a dimensionless factor, recomputed as `(1 + percent / 36000)^(-days)`. [NY Fed publication and revisions](https://www.newyorkfed.org/markets/reference-rates/additional-information-about-reference-rates)

**VALID WITH DOCUMENTED ASSUMPTION — flat SOFR and discount through expiry.** Holding the last available overnight rate flat with daily ACT/360 compounding is a pilot approximation. The Fed's published SOFR Index compounds on business days with simple accrual across non-business days. The baseline also differs from discounting on an observed OIS term curve. Rule 341 establishes the expiry payoff and cash settlement, and the legacy Chapter 341 text in CME filing 21-201 specifies cash settlement on expiration day. Actual settlement-bank value dates and initial premium movements were not verified. Tim owns the choice of curve and exact cash horizon. [CME filing 21-201](https://www.cmegroup.com/market-regulation/rule-filings/2021/6/21-201.pdf)

## Controlled sensitivity

**REQUIRES TIM'S MODEL DECISION — adopting an alternative convention.** The pricer multiplies expected payoff by the supplied discount factor, so the same fitted parameters, paths, and prices can be compared by multiplying each baseline modeled price by the ratio of alternative to baseline factor. The next-business-day scenario uses payment dates September 3, October 1 and November 1, with the August case accounting for Labor Day. The scenarios use no refit or selection on residuals. The simple-interest case holds the same overnight rate and expiry and changes only the accrual formula. The retained experiment uses the original factors.

| Scenario | June RMSE | Mean absolute model-price change | Maximum absolute change |
|---|---:|---:|---:|
| Retained expiry baseline | 0.211321 | 0 | 0 |
| Next-business-day payment | 0.210800 | 0.001323 | 0.016018 |
| Simple ACT/360 to expiry | 0.211256 | 0.000631 | 0.006612 |

The perturbations quantify the effect on the June sample at fixed parameters. July remains reserved for later evaluation.
