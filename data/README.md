# Data lineage and usage

This repository includes the corrected **South German Credit** dataset distributed by the UCI Machine Learning Repository under **CC BY 4.0**.

- Source: <https://archive.ics.uci.edu/dataset/522/south+german+credit>
- DOI: <https://doi.org/10.24432/C5X89F>
- File: `raw/SouthGermanCredit.asc`
- SHA-256: `baa78cca9b7c46631b9d941ed358595b5334e35270b922950e742130617c55f3`
- Grain: one accepted credit contract per row
- Size: 1,000 contracts, 20 predictors, one outcome
- Outcome: 700 good and 300 bad credits

The repository renames the original German headers in `src/config.py` without modifying the source file. `bad_credit=1` is derived from the source label `kredit=0`.

## Evidence boundary

- Contracts are from 1973–1975 and do not represent a current retail portfolio.
- The sample contains accepted applications only; declined applicants are absent.
- Bad credits were intentionally oversampled to 30%.
- `amount` is documented as a transformed historical proxy, so policy values are scenario units rather than currency forecasts.
- `personal_status_sex` combines sex and marital status; sex cannot be reconstructed reliably.
- There is no application timestamp, preventing temporal validation.

The data may be shared and adapted with attribution under the source license. Code in this repository is covered separately by the root MIT license.
