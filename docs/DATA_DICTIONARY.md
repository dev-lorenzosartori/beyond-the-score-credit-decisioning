# Data Dictionary

The public source uses short German column names. The modeling layer renames them to the English fields below.

| Source | Modeling name | Type | Meaning | Treatment |
|---|---|---|---|---|
| `laufkont` | `status` | categorical | Checking-account status | model input |
| `laufzeit` | `duration` | numeric | Credit duration in months | model input |
| `moral` | `credit_history` | categorical | Previous/concurrent credit compliance | model input |
| `verw` | `purpose` | categorical | Credit purpose | model input |
| `hoehe` | `amount` | numeric | Transformed historical credit-amount proxy | model input and scenario exposure proxy |
| `sparkont` | `savings` | categorical | Savings-account band | model input |
| `beszeit` | `employment_duration` | ordinal | Time with current employer | categorical model input |
| `rate` | `installment_rate` | ordinal | Installment share of disposable income | categorical model input |
| `famges` | `personal_status_sex` | categorical | Combined personal-status/sex code | audit-only; unsuitable for sex recovery |
| `buerge` | `other_debtors` | categorical | Co-applicant or guarantor | model input |
| `wohnzeit` | `present_residence` | ordinal | Years at present residence | categorical model input |
| `verm` | `property` | ordinal | Most valuable property category | categorical model input |
| `alter` | `age` | numeric | Age in years | audit-only |
| `weitkred` | `other_installment_plans` | categorical | Other installment-plan provider | model input |
| `wohn` | `housing` | categorical | Housing type | model input |
| `bishkred` | `number_credits` | ordinal | Credits held at this bank | categorical model input |
| `beruf` | `job` | ordinal | Job quality/category | categorical model input |
| `pers` | `people_liable` | binary | Number of financially dependent persons | categorical model input |
| `telef` | `telephone` | binary | Registered landline | categorical model input |
| `gastarb` | `foreign_worker` | binary | Foreign-worker indicator | audit-only |
| `kredit` | `credit_risk` | binary | Contract complied with: 1 good, 0 bad | converted to `bad_credit=1` |

The authoritative level descriptions remain the corrected UCI code table and Grömping (2019). Numeric category codes are never treated as continuous quantities unless the source explicitly documents a quantitative field.
