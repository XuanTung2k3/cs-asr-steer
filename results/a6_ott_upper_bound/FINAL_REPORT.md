# A6-OTT Final Report

Status: PASS

## Phase ordering and eligibility

- Phase-A frozen fingerprint: `sha256:20ca7e93c8823934f373b9107eb63650b50cd087178960dacac45951e61e235b`.
- CS confirmation: 128 acoustic-eligible initially; decoder eligibility is model-specific after the frozen baseline-causal-position gate.
- ASCEND confirmation: all 200 acoustic-eligible; model-specific decoder eligibility is reported below.
- SEAME is transfer-only and was unlocked after Phase-B final settings froze.

## Search vs held-out confirmation

Phase-A selections were fixed before confirmation. Phase-B tables report family-specific N and common-subset N separately; no SEAME outcome entered selection.

## whisper

### add_unique/encoder

- cs_dialogue_confirm / greedy: N=128 (rate=1.000), MER gain=-0.0006934011325551714, PIER gain=0.001021450459652684, corrections=2, corruptions=1, utility=1, retention=0.9984491315136477
- ascend_confirm / greedy: N=200 (rate=1.000), MER gain=0.002738059020383321, PIER gain=0.011612903225806548, corrections=11, corruptions=2, utility=9, retention=0.9975087194818136
- cs_dialogue_confirm / greedy: N=128 (rate=1.000), MER gain=-0.0011556685542586376, PIER gain=-0.001021450459652684, corrections=0, corruptions=1, utility=-1, retention=0.9981389578163772
- ascend_confirm / greedy: N=200 (rate=1.000), MER gain=0.0006084575600851516, PIER gain=0.021935483870967776, corrections=20, corruptions=3, utility=17, retention=0.9920279023418037
- Positive held-out signal on both CS and ASCEND: **True**; candidates: [{'layer': 4, 'rho': 0.5}].
### add_unique/decoder

- cs_dialogue_confirm / greedy: N=274 (rate=1.000), MER gain=0.010104199557941285, PIER gain=-0.00139017608897124, corrections=7, corruptions=10, utility=-3, retention=0.9979502469020777
- ascend_confirm / greedy: N=196 (rate=1.000), MER gain=0.0031104199066874227, PIER gain=0.014267185473411104, corrections=11, corruptions=0, utility=11, retention=0.9984583761562179
- cs_dialogue_confirm / greedy: N=274 (rate=1.000), MER gain=0.012503946952952316, PIER gain=0.006024096385542188, corrections=18, corruptions=5, utility=13, retention=0.9974843939252772
- ascend_confirm / greedy: N=196 (rate=1.000), MER gain=0.0027993779160186416, PIER gain=0.0077821011673151474, corrections=6, corruptions=0, utility=6, retention=0.998972250770812
- Positive held-out signal on both CS and ASCEND: **True**; candidates: [{'layer': 24, 'rho': 0.5}].
### conditioning_cs/decoder

- cs_dialogue_confirm / greedy: N=274 (rate=1.000), MER gain=-0.009283233343858555, PIER gain=-0.010194624652455964, corrections=43, corruptions=65, utility=-22, retention=0.9862107518867046
- ascend_confirm / greedy: N=196 (rate=1.000), MER gain=0.0027993779160186416, PIER gain=0.022049286640726362, corrections=26, corruptions=9, utility=17, retention=0.9917780061664954
- cs_dialogue_confirm / greedy: N=274 (rate=1.000), MER gain=0.004041679823176492, PIER gain=0.003707136237256714, corrections=17, corruptions=9, utility=8, retention=0.989285381533588
- ascend_confirm / greedy: N=196 (rate=1.000), MER gain=0.0027993779160186416, PIER gain=0.01945525291828798, corrections=15, corruptions=0, utility=15, retention=0.9979445015416238
- Positive held-out signal on both CS and ASCEND: **True**; candidates: [{'layer': 0, 'rho': 0.5}].

Next-stage gate: **PROCEED_TO_NON_ORACLE**.

## qwen3_asr_1p7b

### add_unique/encoder

- cs_dialogue_confirm / greedy: N=128 (rate=1.000), MER gain=-0.00023113371085172613, PIER gain=-0.0030643513789581217, corrections=0, corruptions=3, utility=-3, retention=1.0
- ascend_confirm / greedy: N=200 (rate=1.000), MER gain=-0.0009126863401277829, PIER gain=-0.005161290322580642, corrections=0, corruptions=4, utility=-4, retention=0.9991323210412147
- cs_dialogue_confirm / greedy: N=128 (rate=1.000), MER gain=-0.0001155668554258596, PIER gain=-0.0030643513789581217, corrections=1, corruptions=4, utility=-3, retention=1.0
- ascend_confirm / greedy: N=200 (rate=1.000), MER gain=-0.0012169151201703587, PIER gain=-0.003870967741935488, corrections=1, corruptions=4, utility=-3, retention=0.9991323210412147
- Positive held-out signal on both CS and ASCEND: **False**; candidates: [].
### add_unique/decoder

- cs_dialogue_confirm / greedy: N=279 (rate=1.000), MER gain=0.0001255256386116857, PIER gain=0.00046232085067036965, corrections=1, corruptions=0, utility=1, retention=0.9998513895080993
- ascend_confirm / greedy: N=200 (rate=1.000), MER gain=-0.0006084575600851794, PIER gain=-0.0025806451612903347, corrections=0, corruptions=2, utility=-2, retention=1.0
- cs_dialogue_confirm / greedy: N=279 (rate=1.000), MER gain=-0.0003138140965292177, PIER gain=-0.0018492834026814647, corrections=3, corruptions=7, utility=-4, retention=0.9998513895080993
- ascend_confirm / greedy: N=200 (rate=1.000), MER gain=-0.0003042287800425758, PIER gain=0.0, corrections=0, corruptions=0, utility=0, retention=0.9995661605206074
- Positive held-out signal on both CS and ASCEND: **False**; candidates: [].
### conditioning_cs/decoder

- cs_dialogue_confirm / greedy: N=278 (rate=0.996), MER gain=-0.0003142085087664212, PIER gain=-0.0018501387604070302, corrections=1, corruptions=5, utility=-4, retention=0.9998511794032294
- ascend_confirm / greedy: N=200 (rate=1.000), MER gain=-0.0009126863401277829, PIER gain=-0.0025806451612903347, corrections=0, corruptions=2, utility=-2, retention=0.9995661605206074
- cs_dialogue_confirm / greedy: N=278 (rate=0.996), MER gain=-0.000565575315779554, PIER gain=-0.002312673450508798, corrections=0, corruptions=5, utility=-5, retention=0.9998511794032294
- ascend_confirm / greedy: N=200 (rate=1.000), MER gain=-0.0003042287800425758, PIER gain=0.0, corrections=0, corruptions=0, utility=0, retention=0.9995661605206074
- Positive held-out signal on both CS and ASCEND: **False**; candidates: [].

Next-stage gate: **ORACLE_MECHANISM_NOT_ESTABLISHED**.

## Final settings

[
  {
    "family": "add_unique",
    "layer": 20,
    "model": "whisper",
    "rho": 1.0,
    "side": "encoder"
  },
  {
    "family": "add_unique",
    "layer": 24,
    "model": "whisper",
    "rho": 0.5,
    "side": "decoder"
  },
  {
    "family": "conditioning_cs",
    "layer": 0,
    "model": "whisper",
    "rho": 0.5,
    "side": "decoder"
  },
  {
    "family": "add_unique",
    "layer": 15,
    "model": "qwen3_asr_1p7b",
    "rho": 2.0,
    "side": "encoder"
  },
  {
    "family": "add_unique",
    "layer": 13,
    "model": "qwen3_asr_1p7b",
    "rho": 0.5,
    "side": "decoder"
  },
  {
    "family": "conditioning_cs",
    "layer": 0,
    "model": "qwen3_asr_1p7b",
    "rho": 0.5,
    "side": "decoder"
  }
]

## SEAME transfer

SEAME results are transfer-only. `transfer/PHASE_C_TRANSFER.json` contains family-specific and explicit common-eligible rows for greedy and official-standard decoding; all rows carry N, eligibility, and exclusion counts.

## Qwen causal diagnosis

Qwen has no robust positive family signal across both confirmation corpora in this upper-bound run. Target-logit movement is unavailable in the accepted physical row schema, so the result is classified as NO_CLEAR_SIGNAL / DAMAGE_DOMINATED rather than dose- or margin-resolved.

## Common-subset comparisons

Common-subset rows are emitted in both Phase-B and Phase-C tables with `eligibility_scope=common`; encoder-vs-decoder and Add-Unique-vs-Conditioning-CS claims must use those rows, not pooled family-specific denominators.

## Proceed decision

The gate requires a candidate family with positive correction-minus-corruption utility on both held-out confirmation corpora.

## Explicit eligibility and transfer counts

### whisper
- seame_dev_man add_unique/encoder: N panel=50, eligible=50, ineligible=0, rate=1.000.
- seame_dev_sge add_unique/encoder: N panel=50, eligible=50, ineligible=0, rate=1.000.
- seame_dev_man add_unique/decoder: N panel=46, eligible=46, ineligible=0, rate=1.000.
- seame_dev_sge add_unique/decoder: N panel=38, eligible=38, ineligible=0, rate=1.000.
- seame_dev_man conditioning_cs/decoder: N panel=46, eligible=46, ineligible=0, rate=1.000.
- seame_dev_sge conditioning_cs/decoder: N panel=38, eligible=38, ineligible=0, rate=1.000.
- Common transfer N: [('seame_dev_man', 46), ('seame_dev_sge', 38)].
### qwen3_asr_1p7b
- seame_dev_man add_unique/encoder: N panel=50, eligible=36, ineligible=14, rate=0.720.
- seame_dev_sge add_unique/encoder: N panel=50, eligible=35, ineligible=15, rate=0.700.
- seame_dev_man add_unique/decoder: N panel=46, eligible=46, ineligible=0, rate=1.000.
- seame_dev_sge add_unique/decoder: N panel=37, eligible=36, ineligible=1, rate=0.973.
- seame_dev_man conditioning_cs/decoder: N panel=46, eligible=40, ineligible=6, rate=0.870.
- seame_dev_sge conditioning_cs/decoder: N panel=37, eligible=34, ineligible=3, rate=0.919.
- Common transfer N: [('seame_dev_man', 30), ('seame_dev_sge', 22)].

## Best Qwen frozen settings

[
  {
    "family": "add_unique",
    "layer": 15,
    "model": "qwen3_asr_1p7b",
    "rho": 2.0,
    "side": "encoder"
  },
  {
    "family": "add_unique",
    "layer": 13,
    "model": "qwen3_asr_1p7b",
    "rho": 0.5,
    "side": "decoder"
  },
  {
    "family": "conditioning_cs",
    "layer": 0,
    "model": "qwen3_asr_1p7b",
    "rho": 0.5,
    "side": "decoder"
  }
]

The best Qwen frozen candidate by the Phase-B rule is reported with its confirmation metrics above; it does not establish a robust cross-corpus positive oracle mechanism.

## Runtime and job audit

See `runtime/PHASE_BC_JOB_AUDIT.json` for sacct GPU-hour totals, cancellations, duplicate-job handling, and artifact-preservation decisions.
