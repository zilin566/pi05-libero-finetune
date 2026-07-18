# Day 24 Deployment Note

## 1. Evaluation configuration

| Item | Value |
|---|---|
| Checkpoint config | `pi05_libero_day22_bs4_e2` |
| Checkpoint step | `13999` |
| GPU | NVIDIA GeForce RTX 5090 |
| Precision | `bfloat16` |
| Policy batch size | `1` |
| Model internal action dimension | `32` |
| LIBERO environment action dimension | `7` |
| Returned action chunk | `[10, 7]` |
| Replan steps | `5` |

The model uses a unified 32-dimensional internal action
representation. Output transforms remove the padded dimensions before
returning a LIBERO action chunk with shape `[10, 7]`. The rollout
executes five actions from each chunk before requesting another chunk.

## 2. Inference latency

The fresh-server rollout produced `35` policy
queries. The first query was treated as cold start; the remaining
`34` queries were treated as steady state.

| Metric | Result |
|---|---:|
| Cold-start first query | `24850.525 ms` |
| Steady-state mean | `83.246 ms` |
| Steady-state P50 | `81.183 ms` |
| Steady-state P95 | `99.901 ms` |

The approximately 24.85-second first query includes JAX tracing and
compilation and should not be presented as normal online inference
latency. After warm-up, P95 policy-query latency was below 100 ms.

These numbers describe policy-query latency, not complete end-to-end
robot control latency. Image preprocessing, rendering, simulation and
communication may add further delay.

## 3. VRAM

| Metric | Result |
|---|---:|
| GPU baseline | `0 MiB` |
| Loaded-idle resident VRAM | `24617 MiB` (`24.040 GiB`) |
| Observed peak VRAM | `24693 MiB` (`24.114 GiB`) |
| Peak above loaded-idle resident | `76 MiB` |

Most GPU memory was already resident after model loading. The observed
rollout peak was only approximately 76 MiB
above the loaded-idle value.

VRAM was sampled with `nvidia-smi` at approximately 200 ms intervals.
The reported value is therefore an observed sampled peak, not an exact
instantaneous CUDA allocator peak.

## 4. Engineering conclusion

Checkpoint 13999 runs in bf16 with batch size 1 on an RTX 5090. After
JAX warm-up, policy-query latency is approximately
`83.2 ms` on average, with a P95 of
`99.9 ms`. The model requires approximately
`24.04 GiB` of resident VRAM and reached an observed peak
of approximately `24.11 GiB`.

Cold-start compilation is the main deployment latency issue. Persistent
server deployment should therefore warm up the policy before serving
time-sensitive control requests.
