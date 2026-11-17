# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/KaiKaizer7/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/KaiKaizer7/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's launch-day Twitch viewers hour by hour (hours 2–24, UTC), as five log-scale quantiles (5%, 25%, 50%, 75%, 95%) per hour. Practice mean pinball loss 0.2405 vs the house persistence baseline 0.3871 (−37.9%).

## How it works

The program reads one case (`makefun/twitch-ccu-case/v1`) and learns the shape of a launch day at run time from the comparable past launches bundled in the case itself:

- **Level** — the case's own hour-1 print anchors everything.
- **Shape** — for each hour, the log-ratio `ln(v_h) − ln(v_1)` is predicted two ways from the comparables: (a) the median log-ratio of the 8 comparables whose hour-1 print is closest in log space, and (b) a per-hour linear regression of the log-ratio on `ln(hour-1 print)`. The two log-medians are averaged 50/50, floored at the neighbour median.
- **Spread** — log-normal quantiles with sigma = the wider of the two components' dispersions (neighbour IQR, regression residual SD), floored at 0.25 and capped at 1.2.

The key insight: hour 1 (00:00 UTC) catches only the regions where the game has already unlocked, so it is a poor anchor on its own. Launches whose hour 1 is small explode later in the day as more timezones unlock and the US comes online (correlation between `ln(hour-1)` and peak log-growth ≈ −0.94), while launches already huge at hour 1 dip through the UTC daytime and recover into US prime time. The two-component blend captures both regimes.

## Validation

Practice cases p00–p03 (pinball loss on the log scale):

| case | this forecaster | house baseline |
|---|---|---|
| p00 Wolverine | 0.2629 | 0.7437 |
| p01 RDR2 | 0.2286 | 0.2868 |
| p02 GoW Ragnarök | 0.2312 | 0.2146 |
| p03 Spider-Man 2 | 0.2392 | 0.3033 |
| **Mean** | **0.2405** | **0.3871** |

Leave-one-out over the 37 usable bundled comparables: this forecaster 0.4102 vs baseline 0.9297 (−56%). Leave-one-out 90% coverage ≈ 0.93 (honest ≈ 0.90).

## Run it

```sh
MAKEFUN_INPUT_DIR=<dir with *.in> MAKEFUN_OUTPUT_DIR=<out dir> python3 main.py
```

Python 3.13 standard library only. No network, no randomness, fully deterministic.

---

<sub>Entered on [make.fun](https://make.fun/?ref=wdnjyfww&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
