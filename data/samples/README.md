# Actual CME AO source observations

50 real records drawn from the preserved public CME Daily Bulletin (2026-10-26), ClearPort final file (2023-04-03), and official AO EOD sample (2026-10-08). Both calls and puts are included. These are source-backed examples, not the full 5.22-million-row vendor archive. Premium and strike units are USD/barrel; multiplier 1,000 barrels. Null current OI in the ClearPort rows is intentional: the source publishes prior-day interest separately. Definition-enriched expiration dates are labeled and are not independent exchange-document expiry checks.

Original sources:
- https://www.cmegroup.com/daily_bulletin/current/Section63_Energy_Options_Products.pdf
- https://www.cmegroup.com/content/dam/cmegroup/files/download/STLCPC_20230403
- https://datamine.new.cmegroup.com/catalog/dataset/52b3bf08663460e22ca094967f94e6ed

See docs/data for the complete source register, coverage and acquisition workflow. Full licensed MDP-derived archives are not stored in Git.
