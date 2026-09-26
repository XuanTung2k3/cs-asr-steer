# BASIS-A frozen direction comparison

WER is `n/a — no frozen canonical overall WER`; MER is retained as the canonical mixed error metric.

| Direction | rho | source | WER | MER | PIER | Embedded WER | Matrix CER | Corr | Corrupt | Utility | Outside harm | Embed ret. | Matrix ret. | Energy | Utility/Energy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Frozen | 0 | REUSED DG-04 | n/a — no frozen canonical overall WER | 0.262461 | 0.470018 | 0.473545 | 0.2261 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | n/a |
| Local | 0.5 | REUSED DG-04 | n/a — no frozen canonical overall WER | 0.302501 | 0.453263 | 0.535714 | 0.263002 | 135 | 97 | 38 | 1306 | 0.919301 | 0.893027 | 82007.1 | 0.000463375 |
| Local | 1 | REUSED DG-04 | n/a — no frozen canonical overall WER | 1.15481 | 0.720459 | 2.43078 | 0.95358 | 134 | 702 | -568 | 10721 | 0.415973 | 0.119589 | 348818 | -0.00162836 |
| Local | 2 | REUSED DG-04 | n/a — no frozen canonical overall WER | 1.01972 | 0.952822 | 1.29982 | 0.9751 | 13 | 1108 | -1095 | 12104 | 0.078203 | 0.0292923 | 375073 | -0.00291943 |
| Local+Conditioning | 0.5 | REUSED DG-04 | n/a — no frozen canonical overall WER | 0.29656 | 0.474868 | 0.477954 | 0.265071 | 56 | 67 | -11 | 929 | 0.94426 | 0.923875 | 87854.8 | -0.000125207 |
| Local+Conditioning | 1 | REUSED DG-04 | n/a — no frozen canonical overall WER | 1.17792 | 0.676808 | 0.821429 | 1.21396 | 49 | 518 | -469 | 6222 | 0.569052 | 0.495636 | 356372 | -0.00131604 |
| Local+Conditioning | 2 | REUSED DG-04 | n/a — no frozen canonical overall WER | 1.24654 | 0.910053 | 2.29056 | 1.03414 | 44 | 1042 | -998 | 11730 | 0.133111 | 0.056338 | 396685 | -0.00251585 |
| Raw | 0.5 | NEW | n/a — no frozen canonical overall WER | 0.30464 | 0.450617 | 0.532628 | 0.265968 | 137 | 93 | 44 | 1344 | 0.922629 | 0.889657 | 52750.4 | 0.000834117 |
| Raw | 1 | NEW | n/a — no frozen canonical overall WER | 1.11139 | 0.717813 | 2.19841 | 0.940544 | 150 | 712 | -562 | 10853 | 0.407654 | 0.108961 | 150976 | -0.00372245 |
| Raw | 2 | NEW | n/a — no frozen canonical overall WER | 1.01854 | 0.955908 | 1.28395 | 0.975514 | 14 | 1116 | -1102 | 12119 | 0.0715474 | 0.0286011 | 126302 | -0.00872515 |
| Conditioning | 0.5 | NEW | n/a — no frozen canonical overall WER | 0.301848 | 0.548942 | 0.554233 | 0.259277 | 41 | 220 | -179 | 801 | 0.816972 | 0.943576 | 56600.2 | -0.00316254 |
| Conditioning | 1 | NEW | n/a — no frozen canonical overall WER | 0.421196 | 0.700176 | 0.706349 | 0.373845 | 46 | 568 | -522 | 2432 | 0.527454 | 0.826493 | 95159.4 | -0.00548553 |
| Conditioning | 2 | NEW | n/a — no frozen canonical overall WER | 1.1907 | 0.90873 | 2.33642 | 0.984619 | 71 | 1066 | -995 | 10979 | 0.113145 | 0.123563 | 276254 | -0.00360175 |
| Raw+Conditioning | 0.5 | NEW | n/a — no frozen canonical overall WER | 0.298164 | 0.479718 | 0.483245 | 0.266106 | 54 | 76 | -22 | 949 | 0.936772 | 0.922146 | 54555 | -0.000403263 |
| Raw+Conditioning | 1 | NEW | n/a — no frozen canonical overall WER | 1.26383 | 0.668871 | 0.934744 | 1.30432 | 68 | 519 | -451 | 6735 | 0.56822 | 0.451655 | 180320 | -0.00250111 |
| Raw+Conditioning | 2 | NEW | n/a — no frozen canonical overall WER | 1.20282 | 0.900794 | 2.20326 | 1.02683 | 49 | 1026 | -977 | 11736 | 0.146423 | 0.0546963 | 123011 | -0.00794236 |

## Geometry

```json
{
  "direction_order": [
    "raw",
    "local",
    "conditioning",
    "raw_cond",
    "local_cond"
  ],
  "cosine_matrix": [
    [
      1.0,
      0.9962169453590509,
      -0.08690108042758633,
      0.6756844380228146,
      0.6429834143335493
    ],
    [
      0.9962169453590509,
      1.0,
      -2.2551405187698492e-17,
      0.7371909794712582,
      0.7071067811865476
    ],
    [
      -0.08690108042758633,
      -2.2551405187698492e-17,
      1.0,
      0.6756844380228147,
      0.7071067811865477
    ],
    [
      0.6756844380228146,
      0.7371909794712582,
      0.6756844380228147,
      1.0,
      0.9990537886818334
    ],
    [
      0.6429834143335493,
      0.7071067811865476,
      0.7071067811865477,
      0.9990537886818334,
      1.0
    ]
  ],
  "pairwise": {
    "cos_raw_local": 0.9962169453590509,
    "angle_raw_local_degrees": 4.985353351944459,
    "cos_raw_cond": -0.08690108042758633,
    "angle_raw_cond_degrees": 94.98535335194448,
    "cos_local_cond": -2.2551405187698492e-17,
    "angle_local_cond_degrees": 90.0,
    "cos_raw_cond_mixture_local_cond_mixture": 0.9990537886818334,
    "angle_raw_cond_mixture_local_cond_mixture_degrees": 2.492676675972297,
    "cos_raw_cond_mixture_raw": 0.6756844380228146,
    "angle_raw_cond_mixture_raw_degrees": 47.49267667597224,
    "cos_raw_cond_mixture_cond": 0.6756844380228147,
    "angle_raw_cond_mixture_cond_degrees": 47.49267667597223,
    "cos_local_cond_mixture_local": 0.7071067811865476,
    "angle_local_cond_mixture_local_degrees": 45.0,
    "cos_local_cond_mixture_cond": 0.7071067811865477,
    "angle_local_cond_mixture_cond_degrees": 44.999999999999986
  },
  "gram_matrices": {
    "raw": [
      [
        1.0000000000000002,
        -0.08690108042758633
      ],
      [
        -0.08690108042758633,
        1.0000000000000002
      ]
    ],
    "local": [
      [
        1.0000000000000004,
        -1.734723475976807e-17
      ],
      [
        -1.734723475976807e-17,
        1.0000000000000002
      ]
    ]
  },
  "svd": {
    "raw": {
      "singular_values": [
        1.0425454812273593,
        0.9555620961363076
      ],
      "condition_number": 1.0910285008611769,
      "effective_rank": 2
    },
    "local": {
      "singular_values": [
        1.0,
        0.9999999999999999
      ],
      "condition_number": 1.0000000000000002,
      "effective_rank": 2
    }
  },
  "residualization": {
    "alpha_raw_cond": -0.08690108042758633,
    "removed_energy_fraction": 0.007551797779481829,
    "residual_norm_before_renorm": 0.9962169453590509,
    "raw_local_l2_distance": 0.08698338509105191,
    "raw_local_angle_degrees": 4.985353351944459
  },
  "subspace": {
    "principal_angles_degrees": [
      0.0,
      0.0
    ],
    "projection_frobenius_distance": 6.119351641089615e-16,
    "equivalent_within_1e-10": true,
    "raw_projection_matrix": {
      "path": "results/basis_ablation_frozen/geometry/projection_raw.npy",
      "sha256": "sha256:796447b21172847a1ad61c171ce3bac9bce1b612b2554d7a7bacf2f31e4ba754",
      "shape": [
        1280,
        1280
      ]
    },
    "local_projection_matrix": {
      "path": "results/basis_ablation_frozen/geometry/projection_local.npy",
      "sha256": "sha256:6b584b2de0f01c4880d6636c13aa89d450c72ac9bf96f7555b85ae1ffa4c6fe9",
      "shape": [
        1280,
        1280
      ]
    }
  },
  "mixture_coefficients": {
    "a_local": 0.5,
    "a_cond": 0.5
  }
}
```

## PCA limitations

{
  "deferred": false,
  "source": "baseline exact-site L24 teacher-forced states",
  "site": "decoder_post_cross_attn_residual",
  "split": "D-dev-select",
  "sampling_seed": 2408,
  "sampling_policy": "baseline-correct EN/ZH unit query states; max 10000",
  "n_vectors": 10000,
  "dimension": 1280,
  "pc1_explained_variance": 0.05244616837474587,
  "pc2_explained_variance": 0.044118477681064035,
  "pc1_pc2_cumulative": 0.0965646460558099,
  "pc3_explained_variance": 0.03669119973910262,
  "direction_projections": {
    "raw": [
      -0.5186814318410119,
      0.45427056527481735,
      -0.1338955163756528
    ],
    "local": [
      -0.5103810473160963,
      0.44862549078876035,
      -0.13846833288524563
    ],
    "conditioning": [
      0.11773367907828378,
      -0.0844897347064849,
      -0.046593013807471564
    ],
    "raw_cond": [
      -0.29669748939015084,
      0.2736342660564922,
      -0.13355978029571713
    ],
    "local_cond": [
      -0.27764361669600834,
      0.2574828623982676,
      -0.1308581331819351
    ]
  },
  "visual_arrow_scale": 4.0,
  "visual_note": "arrow lengths are multiplied by 4 for readability and are not steering doses"
}

## Projection distributions

{
  "deferred": false,
  "normalization": "featurewise LayerNorm without affine parameters; eps=1e-5",
  "source": "baseline-correct content-unit query states",
  "directions": {
    "raw": {
      "groups": {
        "embedded": {
          "n": 940,
          "mean": 11.561265144361132,
          "std": 4.881531663294397,
          "median": 10.866409111188442,
          "q25": 7.868050290463659,
          "q75": 15.834836408967396
        },
        "matrix": {
          "n": 9060,
          "mean": -2.98762219116386,
          "std": 3.215240723373824,
          "median": -3.4470716087918554,
          "q25": -5.413816298195467,
          "q75": -1.0603313589237933
        }
      },
      "standardized_mean_difference_en_minus_zh": 3.519982990541486
    },
    "local": {
      "groups": {
        "embedded": {
          "n": 940,
          "mean": 11.312783131653985,
          "std": 4.901173813781225,
          "median": 10.554339874851998,
          "q25": 7.551081178220261,
          "q75": 15.646397537754224
        },
        "matrix": {
          "n": 9060,
          "mean": -3.1129525408939056,
          "std": 3.192301481188294,
          "median": -3.565127485977689,
          "q25": -5.514395337902156,
          "q75": -1.2283914029874705
        }
      },
      "standardized_mean_difference_en_minus_zh": 3.4878827210760037
    },
    "conditioning": {
      "groups": {
        "embedded": {
          "n": 940,
          "mean": -3.351844280901888,
          "std": 1.5928804411413942,
          "median": -3.5151533034700675,
          "q25": -4.4669049092228565,
          "q75": -2.364566360362093
        },
        "matrix": {
          "n": 9060,
          "mean": -1.3067027431009457,
          "std": 1.4372275388434583,
          "median": -1.324464312055029,
          "q25": -2.2403476574253682,
          "q75": -0.45301425741279167
        }
      },
      "standardized_mean_difference_en_minus_zh": -1.3481028208553356
    }
  },
  "n_vectors": 10000,
  "language_counts": {
    "EN": 940,
    "ZH": 9060
  }
}

## Dose response

```json
{
  "Frozen": {
    "trajectory": [
      {
        "Direction": "Frozen",
        "rho": 0.0,
        "source": "REUSED DG-04",
        "WER": "n/a \u2014 no frozen canonical overall WER",
        "MER": 0.26246064278500564,
        "PIER": 0.47001763668430335,
        "Embedded WER": 0.47354497354497355,
        "Matrix CER": 0.22610015174506828,
        "Corr": 0,
        "Corrupt": 0,
        "Utility": 0,
        "Outside harm": 0,
        "Embed ret.": 1.0,
        "Matrix ret.": 1.0,
        "Energy": 0.0,
        "Utility/Energy": null,
        "corrections_per_energy": null,
        "corruptions_per_energy": null,
        "corrections_per_1000_edits": null,
        "corruptions_per_1000_edits": null,
        "non_dominated": false
      }
    ],
    "classification": "NO INTERVENTION"
  },
  "Local": {
    "trajectory": [
      {
        "Direction": "Local",
        "rho": 0.5,
        "source": "REUSED DG-04",
        "WER": "n/a \u2014 no frozen canonical overall WER",
        "MER": 0.302501039624547,
        "PIER": 0.4532627865961199,
        "Embedded WER": 0.5357142857142857,
        "Matrix CER": 0.26300179335080703,
        "Corr": 135,
        "Corrupt": 97,
        "Utility": 38,
        "Outside harm": 1306,
        "Embed ret.": 0.9193011647254575,
        "Matrix ret.": 0.8930268728938046,
        "Energy": 82007.07620024681,
        "Utility/Energy": 0.0004633746471732598,
        "corrections_per_energy": 0.0016461994044313177,
        "corruptions_per_energy": 0.001182824757258058,
        "corrections_per_1000_edits": 6.707074721780604,
        "corruptions_per_1000_edits": 4.8191573926868045,
        "non_dominated": false
      },
      {
        "Direction": "Local",
        "rho": 1.0,
        "source": "REUSED DG-04",
        "WER": "n/a \u2014 no frozen canonical overall WER",
        "MER": 1.1548149468306304,
        "PIER": 0.7204585537918872,
        "Embedded WER": 2.4307760141093473,
        "Matrix CER": 0.9535798041109118,
        "Corr": 134,
        "Corrupt": 702,
        "Utility": -568,
        "Outside harm": 10721,
        "Embed ret.": 0.415973377703827,
        "Matrix ret.": 0.11958869783115873,
        "Energy": 348817.9806046486,
        "Utility/Energy": -0.001628356425363786,
        "corrections_per_energy": 0.0003841545088006115,
        "corruptions_per_energy": 0.0020125109341643972,
        "corrections_per_1000_edits": 2.4576333357787394,
        "corruptions_per_1000_edits": 12.875064191915486,
        "non_dominated": false
      },
      {
        "Direction": "Local",
        "rho": 2.0,
        "source": "REUSED DG-04",
        "WER": "n/a \u2014 no frozen canonical overall WER",
        "MER": 1.0197231628349077,
        "PIER": 0.9528218694885362,
        "Embedded WER": 1.2998236331569666,
        "Matrix CER": 0.9751000137950062,
        "Corr": 13,
        "Corrupt": 1108,
        "Utility": -1095,
        "Outside harm": 12104,
        "Embed ret.": 0.07820299500831947,
        "Matrix ret.": 0.02929231832714076,
        "Energy": 375072.8532772064,
        "Utility/Energy": -0.002919432826002778,
        "corrections_per_energy": 3.4659933094096904e-05,
        "corruptions_per_energy": 0.002954092759096875,
        "corrections_per_1000_edits": 0.2877634142022313,
        "corruptions_per_1000_edits": 24.526297148928634,
        "non_dominated": false
      }
    ],
    "classification": "DAMAGE GROWS FASTER THAN CORRECTION"
  },
  "Local+Conditioning": {
    "trajectory": [
      {
        "Direction": "Local+Conditioning",
        "rho": 0.5,
        "source": "REUSED DG-04",
        "WER": "n/a \u2014 no frozen canonical overall WER",
        "MER": 0.2965603279272857,
        "PIER": 0.4748677248677249,
        "Embedded WER": 0.47795414462081126,
        "Matrix CER": 0.2650710442819699,
        "Corr": 56,
        "Corrupt": 67,
        "Utility": -11,
        "Outside harm": 929,
        "Embed ret.": 0.9442595673876872,
        "Matrix ret.": 0.9238745355568997,
        "Energy": 87854.83862447739,
        "Utility/Energy": -0.00012520653582915207,
        "corrections_per_energy": 0.000637415091493865,
        "corruptions_per_energy": 0.0007626216273230171,
        "corrections_per_1000_edits": 2.5725836089672915,
        "corruptions_per_1000_edits": 3.077912532157295,
        "non_dominated": true
      },
      {
        "Direction": "Local+Conditioning",
        "rho": 1.0,
        "source": "REUSED DG-04",
        "WER": "n/a \u2014 no frozen canonical overall WER",
        "MER": 1.1779243153329768,
        "PIER": 0.6768077601410935,
        "Embedded WER": 0.8214285714285714,
        "Matrix CER": 1.2139605462822458,
        "Corr": 49,
        "Corrupt": 518,
        "Utility": -469,
        "Outside harm": 6222,
        "Embed ret.": 0.5690515806988353,
        "Matrix ret.": 0.4956363950574613,
        "Energy": 356371.7889661789,
        "Utility/Energy": -0.0013160413212295824,
        "corrections_per_energy": 0.00013749685445682206,
        "corruptions_per_energy": 0.0014535381756864045,
        "corrections_per_1000_edits": 0.8797127468581688,
        "corruptions_per_1000_edits": 9.299820466786356,
        "non_dominated": false
      },
      {
        "Direction": "Local+Conditioning",
        "rho": 2.0,
        "source": "REUSED DG-04",
        "WER": "n/a \u2014 no frozen canonical overall WER",
        "MER": 1.2465395354363453,
        "PIER": 0.91005291005291,
        "Embedded WER": 2.2905643738977073,
        "Matrix CER": 1.0341426403641882,
        "Corr": 44,
        "Corrupt": 1042,
        "Utility": -998,
        "Outside harm": 11730,
        "Embed ret.": 0.13311148086522462,
        "Matrix ret.": 0.056338028169014086,
        "Energy": 396685.41846227646,
        "Utility/Energy": -0.0025158474538052793,
        "corrections_per_energy": 0.00011091912621987204,
        "corruptions_per_energy": 0.0026267665800251516,
        "corrections_per_1000_edits": 0.9135453865957976,
        "corruptions_per_1000_edits": 21.63441574620048,
        "non_dominated": false
      }
    ],
    "classification": "DAMAGE GROWS FASTER THAN CORRECTION"
  },
  "Raw": {
    "trajectory": [
      {
        "Direction": "Raw",
        "rho": 0.5,
        "source": "NEW",
        "WER": "n/a \u2014 no frozen canonical overall WER",
        "MER": 0.3046396958355611,
        "PIER": 0.4506172839506173,
        "Embedded WER": 0.5326278659611993,
        "Matrix CER": 0.26596771968547384,
        "Corr": 137,
        "Corrupt": 93,
        "Utility": 44,
        "Outside harm": 1344,
        "Embed ret.": 0.9226289517470881,
        "Matrix ret.": 0.8896569601659033,
        "Energy": 52750.38345623016,
        "Utility/Energy": 0.0008341171592905893,
        "corrections_per_energy": 0.002597137518700244,
        "corruptions_per_energy": 0.0017630203594096545,
        "corrections_per_1000_edits": 10.794201071541128,
        "corruptions_per_1000_edits": 7.327450362433028,
        "non_dominated": true
      },
      {
        "Direction": "Raw",
        "rho": 1.0,
        "source": "NEW",
        "WER": "n/a \u2014 no frozen canonical overall WER",
        "MER": 1.11138834432365,
        "PIER": 0.7178130511463845,
        "Embedded WER": 2.1984126984126986,
        "Matrix CER": 0.9405435232445855,
        "Corr": 150,
        "Corrupt": 712,
        "Utility": -562,
        "Outside harm": 10853,
        "Embed ret.": 0.40765391014975044,
        "Matrix ret.": 0.1089605115354705,
        "Energy": 150975.8941373825,
        "Utility/Energy": -0.003722448561812131,
        "corrections_per_energy": 0.0009935360930103552,
        "corruptions_per_energy": 0.004715984654822486,
        "corrections_per_1000_edits": 6.616965900569059,
        "corruptions_per_1000_edits": 31.408531474701135,
        "non_dominated": true
      },
      {
        "Direction": "Raw",
        "rho": 2.0,
        "source": "NEW",
        "WER": "n/a \u2014 no frozen canonical overall WER",
        "MER": 1.0185350204954553,
        "PIER": 0.9559082892416225,
        "Embedded WER": 1.2839506172839505,
        "Matrix CER": 0.9755138639812387,
        "Corr": 14,
        "Corrupt": 1116,
        "Utility": -1102,
        "Outside harm": 12119,
        "Embed ret.": 0.07154742096505824,
        "Matrix ret.": 0.0286010541778277,
        "Energy": 126301.56657075882,
        "Utility/Energy": -0.008725149100843644,
        "corrections_per_energy": 0.00011084581434828587,
        "corruptions_per_energy": 0.00883599491519193,
        "corrections_per_1000_edits": 0.9997857601942441,
        "corruptions_per_1000_edits": 79.69720774119831,
        "non_dominated": false
      }
    ],
    "classification": "DAMAGE GROWS FASTER THAN CORRECTION"
  },
  "Conditioning": {
    "trajectory": [
      {
        "Direction": "Conditioning",
        "rho": 0.5,
        "source": "NEW",
        "WER": "n/a \u2014 no frozen canonical overall WER",
        "MER": 0.3018475613378483,
        "PIER": 0.548941798941799,
        "Embedded WER": 0.5542328042328042,
        "Matrix CER": 0.2592771416747138,
        "Corr": 41,
        "Corrupt": 220,
        "Utility": -179,
        "Outside harm": 801,
        "Embed ret.": 0.8169717138103162,
        "Matrix ret.": 0.9435755638123218,
        "Energy": 56600.16175866127,
        "Utility/Energy": -0.0031625351313171544,
        "corrections_per_energy": 0.0007243795552178957,
        "corruptions_per_energy": 0.00388691468653505,
        "corrections_per_1000_edits": 2.9677886355410785,
        "corruptions_per_1000_edits": 15.924719507781397,
        "non_dominated": false
      },
      {
        "Direction": "Conditioning",
        "rho": 1.0,
        "source": "NEW",
        "WER": "n/a \u2014 no frozen canonical overall WER",
        "MER": 0.42119645933582844,
        "PIER": 0.7001763668430335,
        "Embedded WER": 0.7063492063492064,
        "Matrix CER": 0.3738446682301007,
        "Corr": 46,
        "Corrupt": 568,
        "Utility": -522,
        "Outside harm": 2432,
        "Embed ret.": 0.5274542429284526,
        "Matrix ret.": 0.8264926985224229,
        "Energy": 95159.36358356476,
        "Utility/Energy": -0.00548553479491908,
        "corrections_per_energy": 0.0004833996179430606,
        "corruptions_per_energy": 0.00596893441286214,
        "corrections_per_1000_edits": 3.268670503801606,
        "corruptions_per_1000_edits": 40.360974916506784,
        "non_dominated": false
      },
      {
        "Direction": "Conditioning",
        "rho": 2.0,
        "source": "NEW",
        "WER": "n/a \u2014 no frozen canonical overall WER",
        "MER": 1.1906968454820888,
        "PIER": 0.9087301587301587,
        "Embedded WER": 2.3364197530864197,
        "Matrix CER": 0.9846185680783557,
        "Corr": 71,
        "Corrupt": 1066,
        "Utility": -995,
        "Outside harm": 10979,
        "Embed ret.": 0.11314475873544093,
        "Matrix ret.": 0.1235634666897088,
        "Energy": 276254.3781056404,
        "Utility/Energy": -0.0036017528729246394,
        "corrections_per_energy": 0.00025700950148507476,
        "corruptions_per_energy": 0.0038587623744097146,
        "corrections_per_1000_edits": 2.2379826635145785,
        "corruptions_per_1000_edits": 33.601260835303385,
        "non_dominated": false
      }
    ],
    "classification": "DAMAGE GROWS FASTER THAN CORRECTION"
  },
  "Raw+Conditioning": {
    "trajectory": [
      {
        "Direction": "Raw+Conditioning",
        "rho": 0.5,
        "source": "NEW",
        "WER": "n/a \u2014 no frozen canonical overall WER",
        "MER": 0.2981643200855463,
        "PIER": 0.47971781305114636,
        "Embedded WER": 0.48324514991181655,
        "Matrix CER": 0.2661056697475514,
        "Corr": 54,
        "Corrupt": 76,
        "Utility": -22,
        "Outside harm": 949,
        "Embed ret.": 0.9367720465890182,
        "Matrix ret.": 0.9221463751836171,
        "Energy": 54555.022419691086,
        "Utility/Energy": -0.0004032625966268382,
        "corrections_per_energy": 0.000989826373538603,
        "corruptions_per_energy": 0.001393088970165441,
        "corrections_per_1000_edits": 4.106463878326996,
        "corruptions_per_1000_edits": 5.779467680608365,
        "non_dominated": false
      },
      {
        "Direction": "Raw+Conditioning",
        "rho": 1.0,
        "source": "NEW",
        "WER": "n/a \u2014 no frozen canonical overall WER",
        "MER": 1.2638270064753758,
        "PIER": 0.6688712522045855,
        "Embedded WER": 0.9347442680776014,
        "Matrix CER": 1.3043178369430266,
        "Corr": 68,
        "Corrupt": 519,
        "Utility": -451,
        "Outside harm": 6735,
        "Embed ret.": 0.5682196339434277,
        "Matrix ret.": 0.45165471355741815,
        "Energy": 180320.21135663986,
        "Utility/Energy": -0.0025011062077118233,
        "corrections_per_energy": 0.0003771069226705188,
        "corruptions_per_energy": 0.002878213130382342,
        "corrections_per_1000_edits": 2.548343576675161,
        "corruptions_per_1000_edits": 19.449857592564832,
        "non_dominated": false
      },
      {
        "Direction": "Raw+Conditioning",
        "rho": 2.0,
        "source": "NEW",
        "WER": "n/a \u2014 no frozen canonical overall WER",
        "MER": 1.2028158973445018,
        "PIER": 0.9007936507936508,
        "Embedded WER": 2.20326278659612,
        "Matrix CER": 1.0268312870740792,
        "Corr": 49,
        "Corrupt": 1026,
        "Utility": -977,
        "Outside harm": 11736,
        "Embed ret.": 0.1464226289517471,
        "Matrix ret.": 0.054696275814395574,
        "Energy": 123011.36217927933,
        "Utility/Energy": -0.007942355752276769,
        "corrections_per_energy": 0.0003983371871663886,
        "corruptions_per_energy": 0.008340692939443155,
        "corrections_per_1000_edits": 3.6194415718717683,
        "corruptions_per_1000_edits": 75.78667454572314,
        "non_dominated": false
      }
    ],
    "classification": "DAMAGE GROWS FASTER THAN CORRECTION"
  }
}
```

## Explicit questions

A. Normalized raw/local similarity is cos=0.996216945, angle=4.985353 degrees.
B. Residualization removes 0.007551798 of raw-direction energy; the pre-renormalization residual norm is 0.996216945.
C. Conditioning improves numerical basis conditioning: 1.091028501 to 1.000000000.
D. The rank-2 spans are numerically equivalent: principal angles=[0.0, 0.0] degrees and projection Frobenius distance=6.119e-16.
E. Therefore residualization changes coordinate geometry/conditioning, not the available rank-2 representational subspace.
F. The identical fixed coefficients produce RC/LC cosine=0.999053789, angle=2.492677 degrees.
G. Raw beats Local in utility at rho=.5 and 1.0, while Local beats Raw at rho=2.0; the performance difference is not stable, but the preregistered two-of-three rule labels Raw better.
H. Conditioning alone is not useful on this population: it has negative utility at all doses and more corruptions than corrections.
I. Adding conditioning improves the single-direction utility only at higher doses and loses at rho=.5; the mixture effect is not stable.
J. The correction-damage frontier and non-dominated flags are in frontier.json; the best positive useful-correction/energy point is Raw rho=.5.
K. Raw rho=.5 has the highest positive utility per realized energy; this is not a positive-utility finding at every rho.
L. PCA is supporting only: it uses actual teacher-forced L24 representation rows; projected arrows are descriptive and do not establish causal superiority.
M. Frozen D-dev-select evidence supports retaining the two-vector rationale under the preregistered utility rule, but does not establish learned-controller or held-out generalization evidence.

## Interpretation

All performance conclusions use the full rho trajectories. PCA is descriptive only; it does not establish causal superiority or semantic purity.
