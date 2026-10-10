# BASIS-A3 Cross-Corpus Summary

These observations summarize existing accepted exploratory metrics; they do not repair the scientific snapshot mismatch.

| Observation | CS-Dialogue | SEAME dev-man | SEAME dev-sge |
|---|---|---|---|
| Raw encoder useful region | none at rho=.5; late decoder signal dominates | narrow late Oracle-local L27–L28 PIER response | no positive PIER region; local reduces damage |
| Raw decoder useful region | late, especially L27–L30 Global/Local tradeoff | small late response | small L24/L27 response |
| Raw L27 correction signal | decoder clear; encoder local remains collateral-limited | encoder Oracle-local positive PIER (7 corrections, 3 corruptions) | encoder local has no positive PIER (0 corrections, 7 corruptions) |
| Raw Global→Local rescue | clear on late decoder | strong L27 rescue versus Global, narrow benefit | retention rescue without positive PIER |
| Cond L26 useful | modest, narrow-dose | weak | weak |
| Cond L27 useful | modest Local at rho=1 | weak Local at rho=1 | absent Local |
| Cond Global→Local rescue | strong at L31 | strong at L31 | strong at L31 |
| Cond L31 destructive | yes at rho≥.5 | yes at rho≥.5 | yes at rho≥.5 |
