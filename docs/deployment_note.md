# Deployment Note

> **Project:** openpi / π0.5 + LIBERO  
> **Checkpoint:** `pi05_libero_day22_bs4_e2 / step 13999`  
> **Hardware:** NVIDIA GeForce RTX 5090  
> **Inference precision:** `bfloat16`

---

## 1. Deployment Configuration

| Item | Value |
|---|---|
| Checkpoint config | `pi05_libero_day22_bs4_e2` |
| Checkpoint step | `13999` |
| GPU | NVIDIA GeForce RTX 5090 |
| Inference precision | `bfloat16` |
| Policy batch size | `1` |
| Model internal action dimension | `32` |
| LIBERO environment action dimension | `7` |
| Returned action chunk | `[10, 7]` |
| Replan steps | `5` |

The model uses a unified 32-dimensional internal action representation, while LIBERO consumes 7-dimensional actions.

At inference time, model-side transforms pad model inputs to the internal representation. The policy produces an internal action chunk, after which the output pipeline applies inverse normalization and selects the LIBERO action dimensions:

```text
observation + prompt
→ model-side padding / transforms
→ π0.5 internal action chunk [H, 32]
→ inverse normalization and output transform
→ LIBERO action chunk [H, 7]
```

The rollout executes five actions from each 10-step chunk before requesting a new chunk.

---

## 2. Latency Measurement

The monitored fresh-server rollout produced:

```text
total policy queries      = 35
cold-start queries        = 1
steady-state queries      = 34
```

Results:

| Metric | Result |
|---|---:|
| Cold-start first query | 24,850.525 ms |
| Steady-state mean | 83.246 ms |
| Steady-state P50 | 81.183 ms |
| Steady-state P95 | 99.901 ms |

The first query is excluded from steady-state statistics.

---

## 3. Cold-Start Interpretation

The first query includes:

- framework tracing;
- JIT compilation;
- first-inference initialization.

Therefore:

```text
cold-start first query ≈ 24.85 s
```

must not be mixed with steady-state latency.

Correct reporting:

> After warm-up, the policy-query latency is 83.246 ms on average, with a P95 of 99.901 ms.

The dominant serving overhead is server cold start, not steady-state inference.

---

## 4. VRAM Measurement

| Metric | Result |
|---|---:|
| GPU baseline before server loading | 0 MiB |
| Loaded-idle resident VRAM | 24,617 MiB / 24.040 GiB |
| Observed sampled peak VRAM | 24,693 MiB / 24.114 GiB |
| Peak above resident | 76 MiB |

Most GPU memory was already resident after model loading.

The monitored rollout and compilation stage increased sampled GPU memory by approximately:

```text
76 MiB
```

above the loaded-idle resident value.

---

## 5. Measurement Method

VRAM was sampled using:

```text
nvidia-smi
sampling interval ≈ 200 ms
```

Therefore, the reported peak is:

```text
observed sampled peak VRAM
```

It is not an exact instantaneous CUDA allocator peak. Very short-lived memory spikes may not be captured by the sampling interval.

Latency was measured at the policy-query boundary in the current simulator serving stack.

---

## 6. Runtime Interpretation

With:

```text
action horizon = 10
replan_steps   = 5
```

the policy generates a 10-step action chunk, and the controller executes five actions before replanning.

After warm-up, P95 policy-query latency is below 100 ms.

This number is not a full real-robot end-to-end latency measurement. It excludes:

- sensor exposure and image capture;
- image transport and preprocessing outside the measured boundary;
- network communication not included in the timer;
- safety checks;
- low-level controller delay;
- actuator response.

The measurement should therefore be described as simulator-side policy-query latency.

---

## 7. Recommended Serving Procedure

Recommended serving pattern:

```text
start a persistent Policy Server
→ load checkpoint and normalization assets
→ run at least one warm-up query
→ begin latency-sensitive rollout
→ keep the server resident across episodes
```

The server should not be restarted for every episode.

Recommended monitoring:

```text
record cold-start latency separately
record steady-state query latency
monitor resident and sampled peak VRAM
log runtime exceptions
retain checkpoint and norm-stats checksums
```

---

## 8. Engineering Conclusion

Checkpoint 13,999 runs successfully with:

```text
precision          = bfloat16
batch size         = 1
returned chunk     = [10, 7]
replan steps       = 5
```

Steady-state measurements:

```text
mean latency       = 83.246 ms
P50 latency        = 81.183 ms
P95 latency        = 99.901 ms
resident VRAM      = 24.040 GiB
observed peak      = 24.114 GiB
```

Recommended project statement:

> The π0.5 step-13,999 policy runs in bf16 on an RTX 5090 with an average steady-state policy-query latency of 83.2 ms, a P95 of 99.9 ms, approximately 24.04 GiB of loaded-idle resident VRAM, and an observed sampled peak of 24.11 GiB. The dominant deployment overhead is the approximately 24.85-second cold start.

---

## 9. Limitations

1. Latency was measured on one RTX 5090 system.
2. Policy batch size was fixed at 1.
3. The primary engineering record used one monitored fresh-server rollout.
4. VRAM was sampled with `nvidia-smi`, not measured from an exact allocator trace.
5. The result is simulator-side policy-query latency, not real-robot end-to-end latency.
6. No quantization, TensorRT-style optimization, or deployment-specific kernel tuning was applied.
7. The measurement does not establish performance on other GPUs or serving stacks.

---

## 10. Evidence Files

```text
experiments/day24_deployment/deployment_note.md
experiments/day24_deployment/latency_vram.csv
experiments/day24_deployment/monitored_fresh_rollout/deployment_measurement.json
notes/day24_notes.md
```
