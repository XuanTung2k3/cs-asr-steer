# Six-GPU execution version v2

Owner steering: prioritize N1/N2/N3 in approximately seven hours, then confirmation. Up to all six idle
A5000 GPUs are authorized. This supersedes the single-GPU operational limit; scientific recipes are unchanged.

The v1 run was safely stopped during B0 (80 predictions, no adaptation trial). Its349.591334 seconds,
including admission, are charged against the same72 aggregate GPU-hour ceiling. Its immutable output remains.

New branch exp/a5000-energy-campaign-multigpu-v2 uses an arm-level DAG. Each trial stays on one GPU;
no distributed gradient synchronization, batch-size scaling or LR scaling. Six independent processes can
run concurrently. Selection jobs execute on CPU as soon as their own two trials finish. Gate jobs start
from the selected CE ReFT even while unrelated baseline jobs run. CE/WCE cells are independent. N2 NO-GO
skips gated-WCE expansion while the ungated WCE matrix continues. N4/N5 start after the full core report.

Frozen scientific fields training/gate/data/pilot_rule/decode/metrics/confirmation exactly match v1.
A shared immutable frozen-encoder cache uses per-utterance process locks and atomic writes. Weight-LoRA
always recomputes its adapted encoder. Each job has separate logs, optimizer checkpoints, RNG/cursor state,
source/config/data identities and status. Controller GPU locks prevent duplicate campaign ownership; busy
GPUs owned by others are excluded. Same RTX A5000 class/precision is checked across available devices.

Planning uses actual warm forward/backward timings and B0's80-utterance mean decode1.7242s. With a1.5
storage/checkpoint safety factor, the deterministic DAG estimate is:

|Concurrent GPUs|Core wall-hours|Core aggregate GPU-hours|
|---|---:|---:|
|1|26.19|26.19|
|2|13.45|26.19|
|3|9.56|26.19|
|4|7.03|26.19|
|6|5.06|26.19|

Practical target:4–7 hours after launch for N1/N2/N3, with extra margin up to8h. This is an estimate,
not permission to truncate a competitor or a promise that all N0–N5 finish in7h. Actual ETA depends on
whole-panel generation lengths, shared storage, available GPUs and conditional gate work. The controller
keeps running valid downstream confirmation afterward, within the unchanged72GPUh/120wallh caps and3GPUh
reserve. Scientific negative outcomes do not trigger architecture/hyperparameter changes.

Tmux survives SSH/Mac disconnects. One bounded controller infrastructure restart waits for surviving
workers before resuming. Reboot recovery requires explicit resume. No automatic system boot service is claimed.
Live state is in the immutable v2 run root, including job_graph.json/resource_estimate.json/status.json,
CAMPAIGN_STATUS.md, campaign.log, logs/<job>.log, jobs/<job>/training/checkpoints and final reports.
