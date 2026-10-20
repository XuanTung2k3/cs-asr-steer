# Frozen execution graph

|Stage|Dependencies|Outputs|Eligibility|
|---|---|---|---|
|N0|committed source, original TRAIN manifest/model|data_manifest.json, preflight/report.json, cpu-tests.xml, admitted.json|all admission checks|
|N1|N0|N1_selection.json; B0; 4methods ×2LR ×3epoch checkpoints/predictions|required|
|N2|N1|scalar panel; G3/G4/G5/G8; N2_decision.json|required|
|N3|N1|4WCE methods; optional G5 WCE|independent of GO/NO-GO|
|N4|N1,N2,N3|bounded D1/D2; G2/G6/G7/G9/gatedLRA if GO|D3 explicitly blocked|
|N5|N1–N4|independent decisive seeds; confirmation_freeze.json; C500 package|remaining budget, no missing dependencies|
|REPORT|all available evidence|CSV, LaTeX, PDFs, Pareto and limitations|always, including NO-GO/resource stop|

Partial arms are not scientific comparisons. N1 is not complete without all four comparators. Optional final official/external testing never opens automatically.
