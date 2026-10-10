# BASIS-A4 Storage Recovery

Recorded before cleanup: 2026-09-11.

```text
Before: root-backed shared PVC 10T, 9.7T used, 329G available, 97%
After:  root-backed shared PVC 10T, 9.7T used, 337G available, 97%
```

The root filesystem is a shared approximately 10 TB mount; deleting this
user's files cannot reduce the shared percentage below 85%. The safe cleanup
was nevertheless performed before further GPU work:

- `/home/tungnx/miniconda3/pkgs`: conda package/tarball cache reduced from
  approximately 16G to 7.5G using `conda clean --all --yes`.
- No accepted A3/A4 results, manifests, provenance, model files, or Slurm
  audit logs were deleted or moved.
- The Qwen model and data are on `/mnt/data`; no duplicate model download was
  found on the root-backed filesystem.

The failed/invalid A4 output was retained and quarantined separately under
`results/basis_a4/quarantine/`.
