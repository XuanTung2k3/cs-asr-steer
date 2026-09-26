# BASIS-A5 Unique–Shared Local Steering Atlas

## Status

COMPLETE for the exploratory atlas. The preserved construction-only rank gate remains a recorded `FAIL` and is not relabeled as a pass. Under the explicit amendment, rank stability is diagnostic and nonblocking.

## Direction construction

Directions were constructed from uncentered D-construct moments at the exact A4 sites. Signs were fixed from construction means only. S1 is Minus-Shared, S2 is Add-Unique, and S3 is the Unique-minus-Shared composite; S3 is not a difference-in-means direction.

## Scientific interpretation

The following aggregate metrics are descriptive and do not force a positive conclusion:

```json
{
  "qwen3_asr_1p7b:neg_shared": {
    "matrix_retention": 0.9942031366069003,
    "mer": 0.2011037968772697,
    "pier": 0.509844450816673,
    "poi_net_utility": -3.891025641025641
  },
  "qwen3_asr_1p7b:pos_unique": {
    "matrix_retention": 0.9971914001844454,
    "mer": 0.19915776939579244,
    "pier": 0.5097542632264855,
    "poi_net_utility": -1.5
  },
  "qwen3_asr_1p7b:unique_minus_shared": {
    "matrix_retention": 0.9963270807565069,
    "mer": 0.19940758112057283,
    "pier": 0.5099826870660203,
    "poi_net_utility": -2.121794871794872
  },
  "whisper:neg_shared": {
    "matrix_retention": 0.8941321377446408,
    "mer": 0.40247638460231544,
    "pier": 0.7391104747875582,
    "poi_net_utility": -131.17708333333334
  },
  "whisper:pos_unique": {
    "matrix_retention": 0.9574688072285826,
    "mer": 0.3548200346384813,
    "pier": 0.6736535953115815,
    "poi_net_utility": -23.994791666666668
  },
  "whisper:unique_minus_shared": {
    "matrix_retention": 0.9077892277424082,
    "mer": 0.39038333680948617,
    "pier": 0.7178472890278446,
    "poi_net_utility": -95.91666666666667
  }
}
```

The accepted table supports direct answers to the A5 questions through the per-cell MER, PIER, retention, correction, corruption, energy, and comparator columns. Any high corrective power paired with damage is reported as such. A5 is Oracle-local and therefore an upper-bound mechanism test, not a deployable localizer.

## Stable versus unstable analysis

The following summaries are descriptive. `UNSTABLE` means the preserved rank diagnostic was below its stability threshold; `NOT_TESTED` means the old anchor diagnostic did not cover that layer. No result was removed because of this label, and no result was used to choose rank.

```json
{
  "by_group": {
    "qwen3_asr_1p7b:decoder:neg_shared": {
      "a5_minus_conditioning_local_mer": 0.0031018056763972898,
      "a5_minus_raw_local_mer": 0.0032988322692101193,
      "en_wer": 0.5264612298342456,
      "matrix_cer": null,
      "matrix_retention": 0.9954037302019524,
      "mer": 0.20097726636435492,
      "pier": 0.507158699619017,
      "poi_corrections": 0.2261904761904762,
      "poi_corruptions": 5.25,
      "poi_net_utility": -5.023809523809524,
      "total_intervention_energy": 38181.681868615604
    },
    "qwen3_asr_1p7b:decoder:pos_unique": {
      "a5_minus_conditioning_local_mer": -0.00035209022714145636,
      "a5_minus_raw_local_mer": -0.00015506363432862626,
      "en_wer": 0.524550111851699,
      "matrix_cer": null,
      "matrix_retention": 0.9998946711296942,
      "mer": 0.1975233704608162,
      "pier": 0.505394554005665,
      "poi_corrections": 0.13095238095238096,
      "poi_corruptions": 0.8928571428571429,
      "poi_net_utility": -0.7619047619047619,
      "total_intervention_energy": 20155.003940375078
    },
    "qwen3_asr_1p7b:decoder:unique_minus_shared": {
      "a5_minus_conditioning_local_mer": -7.982849228516274e-05,
      "a5_minus_raw_local_mer": 0.00011719810052766737,
      "en_wer": 0.5249860185376056,
      "matrix_cer": null,
      "matrix_retention": 0.9994301906644139,
      "mer": 0.19779563219567248,
      "pier": 0.5058304606915718,
      "poi_corrections": 0.14285714285714285,
      "poi_corruptions": 1.6666666666666667,
      "poi_net_utility": -1.5238095238095237,
      "total_intervention_energy": 30377.180345336597
    },
    "qwen3_asr_1p7b:encoder:neg_shared": {
      "a5_minus_conditioning_local_mer": null,
      "a5_minus_raw_local_mer": -0.0011118888524388074,
      "en_wer": 0.5332945304010119,
      "matrix_cer": null,
      "matrix_retention": 0.9928024440793397,
      "mer": 0.20125141580900355,
      "pier": 0.5129778272139383,
      "poi_corrections": 0.8611111111111112,
      "poi_corruptions": 3.4305555555555554,
      "poi_net_utility": -2.5694444444444446,
      "total_intervention_energy": 24278.212916586133
    },
    "qwen3_asr_1p7b:encoder:pos_unique": {
      "a5_minus_conditioning_local_mer": null,
      "a5_minus_raw_local_mer": -0.0012987365081777204,
      "en_wer": 0.5346874276272424,
      "matrix_cer": null,
      "matrix_retention": 0.9940375840816554,
      "mer": 0.20106456815326462,
      "pier": 0.5148405906507758,
      "poi_corrections": 0.7083333333333334,
      "poi_corruptions": 3.0694444444444446,
      "poi_net_utility": -2.361111111111111,
      "total_intervention_energy": 22800.337412463294
    },
    "qwen3_asr_1p7b:encoder:unique_minus_shared": {
      "a5_minus_conditioning_local_mer": null,
      "a5_minus_raw_local_mer": -0.001075116461819116,
      "en_wer": 0.5346737881460103,
      "matrix_cer": null,
      "matrix_retention": 0.9927067858639487,
      "mer": 0.20128818819962324,
      "pier": 0.5148269511695437,
      "poi_corrections": 0.8472222222222222,
      "poi_corruptions": 3.6666666666666665,
      "poi_net_utility": -2.8194444444444446,
      "total_intervention_energy": 25634.32245710161
    },
    "whisper:decoder:neg_shared": {
      "a5_minus_conditioning_local_mer": 0.015170664755515774,
      "a5_minus_raw_local_mer": 0.025914536629633902,
      "en_wer": 0.7164744334883224,
      "matrix_cer": null,
      "matrix_retention": 0.9756284096574301,
      "mer": 0.33323062401959525,
      "pier": 0.6994392085537919,
      "poi_corrections": 15.416666666666666,
      "poi_corruptions": 87.03125,
      "poi_net_utility": -71.61458333333333,
      "total_intervention_energy": 2413.652208507061
    },
    "whisper:decoder:pos_unique": {
      "a5_minus_conditioning_local_mer": -0.011974272747711479,
      "a5_minus_raw_local_mer": -0.00123040087359335,
      "en_wer": 0.6619378724413447,
      "matrix_cer": null,
      "matrix_retention": 0.9945696704801671,
      "mer": 0.30608568651636797,
      "pier": 0.6489483542434932,
      "poi_corrections": 20.21875,
      "poi_corruptions": 9.3125,
      "poi_net_utility": 10.90625,
      "total_intervention_energy": 1845.6826236030709
    },
    "whisper:decoder:unique_minus_shared": {
      "a5_minus_conditioning_local_mer": -0.009186170527584695,
      "a5_minus_raw_local_mer": 0.0015577013465334353,
      "en_wer": 0.6773263387846722,
      "matrix_cer": null,
      "matrix_retention": 0.9924947197588415,
      "mer": 0.3088737887364948,
      "pier": 0.6660935996873497,
      "poi_corrections": 18.1875,
      "poi_corruptions": 36.104166666666664,
      "poi_net_utility": -17.916666666666668,
      "total_intervention_energy": 2109.7316060693315
    },
    "whisper:encoder:neg_shared": {
      "a5_minus_conditioning_local_mer": null,
      "a5_minus_raw_local_mer": -0.11128620069568089,
      "en_wer": 0.7880301443669498,
      "matrix_cer": null,
      "matrix_retention": 0.8126358658318514,
      "mer": 0.4717221451850357,
      "pier": 0.7787817410213242,
      "poi_corrections": 15.177083333333334,
      "poi_corruptions": 205.91666666666666,
      "poi_net_utility": -190.73958333333334,
      "total_intervention_energy": 278397.9719959398
    },
    "whisper:encoder:pos_unique": {
      "a5_minus_conditioning_local_mer": null,
      "a5_minus_raw_local_mer": -0.1794539631201221,
      "en_wer": 0.7173748480172092,
      "matrix_cer": null,
      "matrix_retention": 0.9203679439769982,
      "mer": 0.4035543827605945,
      "pier": 0.6983588363796697,
      "poi_corrections": 28.302083333333332,
      "poi_corruptions": 87.19791666666667,
      "poi_net_utility": -58.895833333333336,
      "total_intervention_energy": 220422.72609479228
    },
    "whisper:encoder:unique_minus_shared": {
      "a5_minus_conditioning_local_mer": null,
      "a5_minus_raw_local_mer": -0.11111546099823925,
      "en_wer": 0.7902347299048688,
      "matrix_cer": null,
      "matrix_retention": 0.8230837357259748,
      "mer": 0.47189288488247744,
      "pier": 0.7696009783683394,
      "poi_corrections": 16.270833333333332,
      "poi_corruptions": 190.1875,
      "poi_net_utility": -173.91666666666666,
      "total_intervention_energy": 307749.2715725402
    }
  },
  "cross_corpus": {
    "cs_dialogue:neg_shared": {
      "mer": 0.24759657241420285,
      "n": 116,
      "pier": 0.39417229216079797,
      "poi_net_utility": -218.8793103448276
    },
    "cs_dialogue:pos_unique": {
      "mer": 0.19990955778571237,
      "n": 116,
      "pier": 0.3157612661923007,
      "poi_net_utility": -41.043103448275865
    },
    "cs_dialogue:unique_minus_shared": {
      "mer": 0.23049858959310224,
      "n": 116,
      "pier": 0.36776287782034917,
      "poi_net_utility": -158.98275862068965
    },
    "seame_dev_man:neg_shared": {
      "mer": 0.3534392489619065,
      "n": 116,
      "pier": 0.6893068617206549,
      "poi_net_utility": -1.5862068965517242
    },
    "seame_dev_man:pos_unique": {
      "mer": 0.3424264307636758,
      "n": 116,
      "pier": 0.6765935214211077,
      "poi_net_utility": -0.3275862068965517
    },
    "seame_dev_man:unique_minus_shared": {
      "mer": 0.35715833182885,
      "n": 116,
      "pier": 0.6812957157784745,
      "poi_net_utility": -0.7931034482758621
    },
    "seame_dev_sge:neg_shared": {
      "mer": 0.33558192135232723,
      "n": 116,
      "pier": 0.8255289968652038,
      "poi_net_utility": -1.8879310344827587
    },
    "seame_dev_sge:pos_unique": {
      "mer": 0.3127852069362327,
      "n": 116,
      "pier": 0.8081896551724138,
      "poi_net_utility": -0.3620689655172414
    },
    "seame_dev_sge:unique_minus_shared": {
      "mer": 0.3266636244593468,
      "n": 116,
      "pier": 0.8249412225705328,
      "poi_net_utility": -1.8362068965517242
    }
  },
  "cross_model": {
    "decoder:neg_shared": {
      "qwen_mer": 0.20097726636435492,
      "qwen_pier": 0.507158699619017,
      "whisper_mer": 0.33323062401959525,
      "whisper_pier": 0.6994392085537919
    },
    "decoder:pos_unique": {
      "qwen_mer": 0.1975233704608162,
      "qwen_pier": 0.505394554005665,
      "whisper_mer": 0.30608568651636797,
      "whisper_pier": 0.6489483542434932
    },
    "decoder:unique_minus_shared": {
      "qwen_mer": 0.19779563219567248,
      "qwen_pier": 0.5058304606915718,
      "whisper_mer": 0.3088737887364948,
      "whisper_pier": 0.6660935996873497
    },
    "encoder:neg_shared": {
      "qwen_mer": 0.20125141580900355,
      "qwen_pier": 0.5129778272139383,
      "whisper_mer": 0.4717221451850357,
      "whisper_pier": 0.7787817410213242
    },
    "encoder:pos_unique": {
      "qwen_mer": 0.20106456815326462,
      "qwen_pier": 0.5148405906507758,
      "whisper_mer": 0.4035543827605945,
      "whisper_pier": 0.6983588363796697
    },
    "encoder:unique_minus_shared": {
      "qwen_mer": 0.20128818819962324,
      "qwen_pier": 0.5148269511695437,
      "whisper_mer": 0.47189288488247744,
      "whisper_pier": 0.7696009783683394
    }
  },
  "method_names": {
    "neg_shared": "Minus-Shared",
    "pos_unique": "Add-Unique",
    "unique_minus_shared": "Unique-minus-Shared"
  },
  "questions": {
    "Q10_unique_beyond_raw": "Compare pos_unique versus Raw-local and unique_minus_shared versus Raw-local across layers/corpora.",
    "Q1_plus_unique": {
      "descriptive_delta": -0.02716551574760523,
      "mean_mer": 0.28504039849520696,
      "reference_minus_shared_mer": 0.3122059142428122
    },
    "Q2_minus_shared": "Inspect its MER/PIER and matrix retention; a lower MER with lower retention is not a clean benefit.",
    "Q3_us_vs_components": {
      "descriptive_delta": -0.007432398949045904,
      "minus_shared_mean_mer": 0.3122059142428122,
      "us_mean_mer": 0.3047735152937663
    },
    "Q4_plus_unique_vs_raw": "Use a5_minus_raw_local_mer for pos_unique; negative is lower MER.",
    "Q5_us_vs_raw": "Use a5_minus_raw_local_mer for unique_minus_shared; negative is lower MER.",
    "Q6_encoder_vs_decoder": "Compare by_group entries with the same method; no side is selected by protocol.",
    "Q7_cross_model": "Compare relative-depth figure and cross_model table; this is descriptive, not a transfer claim.",
    "Q8_cross_corpus": "Compare cross_corpus entries; sign consistency is required before describing transfer.",
    "Q9_rank_instability": "Compare stability entries, including n and mer_std; one favorable unstable layer is insufficient."
  },
  "stability": {
    "NOT_TESTED": {
      "matrix_retention": 0.9560189462760668,
      "mer": 0.2928718566895986,
      "mer_std": 0.14080081482784193,
      "n": 945,
      "pier": 0.616138308422259,
      "poi_corrections": 10.337566137566137,
      "poi_corruptions": 54.72380952380952,
      "poi_net_utility": -44.386243386243386
    },
    "STABLE": {
      "matrix_retention": 0.992544217919574,
      "mer": 0.26804435599284016,
      "mer_std": 0.10362491888205524,
      "n": 63,
      "pier": 0.6034819090374645,
      "poi_corrections": 11.523809523809524,
      "poi_corruptions": 29.174603174603174,
      "poi_net_utility": -17.650793650793652
    },
    "UNSTABLE": {
      "matrix_retention": 0.8310915427513884,
      "mer": 0.5625611432178265,
      "mer_std": 0.17409891104948128,
      "n": 36,
      "pier": 0.7617227165838277,
      "poi_corrections": 17.333333333333332,
      "poi_corruptions": 193.0,
      "poi_net_utility": -175.66666666666666
    }
  }
}
```

Whisper/Qwen comparisons use relative depth only. Hidden coordinates are not compared across models, and rho=.5 is not treated as energy-matched.

## Completeness

Whisper: 576/576. Qwen: 468/468. Total: 1044/1044. Duplicate keys: 0. Empty provenance: 0. Conditioning-Avg: not part of A5.

## Explicit answers to the A5 questions

1. **Does Add-Unique help embedded-language recognition?** Descriptively, yes
   on average: mean MER is lower than the Minus-Shared reference (0.2850 vs
   0.3122). The clearest useful effect is Whisper decoder-side Add-Unique,
   which has positive POI net utility; Qwen effects are small and its net
   utility remains slightly negative. This is not a claim of harmlessness.

2. **Does Minus-Shared help, or mostly remove useful common information?** It
   mostly removes useful information in Whisper: it has the lowest retention
   and strongly negative POI utility, especially in the encoder. Qwen is much
   less affected, but Minus-Shared is not a clean benefit there either.

3. **Does U-S outperform either component?** U-S improves mean MER over
   Minus-Shared (0.3048 vs 0.3122), but does not outperform Add-Unique
   (0.2850). Its extra shared subtraction also increases damage relative to
   Add-Unique in several panels.

4. **Does Add-Unique outperform Raw-local?** Often on MER, particularly in
   Whisper encoder and decoder summaries, and it is better than Raw-local in
   the Qwen decoder summary. The improvement is not uniformly beneficial:
   encoder corrections can come with substantial corruptions and negative
   net utility.

5. **Does U-S outperform Raw-local?** Not consistently. It improves the
   Whisper encoder summary and is close to Raw in Qwen, but is worse than Raw
   in the Whisper decoder summary and does not establish a general advantage.

6. **Are improvements mainly decoder-side?** The most useful tradeoff is
   decoder-side Add-Unique in Whisper. Encoder Add-Unique can lower error
   rates but causes much larger intervention damage; Qwen shows only small
   encoder/decoder separation. Thus the answer is mainly decoder-side for
   useful, damage-aware behavior, not for raw error reduction alone.

7. **Do Whisper and Qwen show similar relative-depth behavior?** They show a
   similar method ordering in aggregate—Add-Unique is best, U-S is close
   behind, and Minus-Shared is worst—but the effect sizes differ greatly.
   Relative-depth figures therefore support a descriptive resemblance, not a
   cross-model transfer claim.

8. **Do signs/patterns transfer from CS to SEAME?** Partially. Add-Unique
   lowers MER relative to Minus-Shared on both SEAME panels and CS, while
   U-S is better than Minus-Shared on CS and SEAME-sge but slightly worse on
   SEAME-man. Net utility remains mostly negative, so transfer is not clean.

9. **Does rank instability correlate with worse or noisier behavior?** In this
   atlas, yes descriptively: unstable layers have MER 0.563, MER standard
   deviation 0.174, retention 0.831, and POI net utility -175.7, versus
   stable-layer MER 0.268, standard deviation 0.104, retention 0.993, and
   utility -17.7. This correlation is post-hoc and does not make instability a
   rank-selection criterion; untested layers remain a separate category.

10. **Does Unique provide anything beyond Raw?** Yes, but conditionally.
    Add-Unique can beat Raw-local in MER and gives the clearest positive
    decoder-side Whisper utility. U-S does not add a consistent advantage
    beyond Raw. The result supports Unique as a causal exploratory component,
    not as a universally superior replacement for Raw-local.

## Figures and tables

Required figures are under `figures/`; accepted rows are in `tables/basis_a5_atlas.csv`; within-model direction geometry is in `geometry/a5_vs_a4.csv`.

## Data exposure

Vectors use D-construct only. Evaluation uses the frozen A4 D-dev-select, dev-man, and dev-sge panels. D-dev-confirm and D-test were not read or intervened.

## Gate

READY_FOR_INDEPENDENT_AUDIT
