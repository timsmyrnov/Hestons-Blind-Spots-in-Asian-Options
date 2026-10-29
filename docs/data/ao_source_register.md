# AO source register

Original files, raw observations, timestamps, and validation evidence remain separate. The full manifest is `evidence/source_manifest_extended.csv`; collection receipts preserve request parameters and completion times. A source website advertising history is not treated as proof of downloaded coverage.

- **[CME AO rule](https://www.cmegroup.com/rulebook/NYMEX/3/341.pdf)** — Legal current AO specification; historical version changes not exhaustive. Access result: web verified; direct PDF retrieval timed out.
- **[CME CL rule](https://www.cmegroup.com/rulebook/NYMEX/2/200.pdf)** — Expiry and contract specification. Access result: web verified; direct PDF retrieval timed out.
- **[CME CSX rule](https://www.cmegroup.com/rulebook/NYMEX/5/510.pdf)** — Arithmetic nearby average independent validation reference. Access result: web verified; direct PDF retrieval timed out.
- **[Bachelier notice](https://www.cmegroup.com/notices/clearing/2020/04/Chadv20-171.pdf)** — Effective Apr22 2020 EOD; includes AO/AAO. Access result: web verified; direct PDF retrieval timed out.
- **[Curran notice](https://www.cmegroup.com/notices/clearing/2020/05/Chadv20-198.pdf)** — Effective May18 2020 EOD; includes AO/AAO. Access result: web verified; direct PDF retrieval timed out.
- **[AO Daily Bulletin](https://www.cmegroup.com/daily_bulletin/current/Section63_Energy_Options_Products.pdf)** — 2026-10-26 2397 AO observations; 66 OI totals. Access result: downloaded original PDF.
- **[AO ClearPort sample](https://www.cmegroup.com/content/dam/cmegroup/files/download/STLCPC_20230403)** — 2023-04-03 1579 AO observations; prior-day OI distinguished. Access result: downloaded original ASCII.
- **[CL final sample](https://www.cmegroup.com/content/dam/cmegroup/files/download/STLNYMEX_20230403)** — 2023-04-03 expiry-specific CL prices; no AO in this file. Access result: downloaded original ASCII.
- **[AO EOD product/sample](https://datamine.new.cmegroup.com/catalog/dataset/52b3bf08663460e22ca094967f94e6ed)** — EOD_XNYM_AO_OPT_0; downloaded 10-row 2026-10-08 sample; 2024-25 quote $1032, 2023-25 $1548. Access result: portal verified; no purchase.
- **[AAO Statistics](https://databento.com/catalog/cme/GLBX.MDP3/options/AAO)** — 2020-2026 actual downloaded archive; not just advertised coverage. Access result: downloaded 71059134 raw messages; $5.2943180501461.
- **[Statistics schema](https://databento.com/docs/schemas/statistics)** — Settlement, cleared volume, OI, action and publication/reference timestamps. Access result: official documentation verified.
- **[GLBX flags](https://databento.com/docs/venues-and-datasets/glbx-mdp3)** — Dataset-specific settlement flags and dataset restrictions. Access result: official documentation.
- **[FTP retirement](https://www.cmegroup.com/notices/clearing/2023/09/Chadv23-272.pdf)** — Jan2024 removal; historical settlement access via DataMine. Access result: official notice; free historical bulletin not established.
- **[Bulletin archive guidance](https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457420037/Daily+Bulletin)** — Current bulletin availability; historical requests route to DataMine. Access result: official documentation.
- **[CME licensing support](https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457319709/CME+DataMine+Support)** — DataMine sample and student support; no student discount quote claimed. Access result: official support.

AO originals: 2,397 bulletin rows, 1,579 ClearPort rows and 10 EOD sample rows. All 3,986 overlapping settlement prices match the vendor archive. Independent current-OI comparisons: 10/10; volume: 83/83. Only the EOD sample independently publishes expiration; other free-source expirations were definition enriched.

The current bulletin overwrites, and no arbitrary-date free 2020 archive was retrieved. Exact paid fallback quotes are preserved in the evidence folder. Bachelier/Curran source notices are verified official web PDFs; direct binary downloads timed out and are explicitly logged. The current live extraction code reads actual local originals and fails on wrong dates, layout/field errors, or failed OI totals.


## October 29 model-input additions

The official NY Fed SOFR search endpoint supplied 1,742 dated observations for November 2019–October 2026. Raw JSON and its checksum/collection receipt remain in the local source archive. CME's 2021 and 2023 Good Friday notices explicitly document copied NYMEX marks; these were read through official web documents, while direct binary retrieval attempts timed out. The updated reconciliation table and model defaults describe the resulting exclusions and remaining limitations.

- https://www.newyorkfed.org/markets/reference-rates/sofr
- https://www.cmegroup.com/tools-information/holiday-calendar/files/good-friday-holiday-settlement-times-2021.pdf
- https://www.cmegroup.com/tools-information/holiday-calendar/files/good-friday-holiday-settlement-times-2023.pdf
