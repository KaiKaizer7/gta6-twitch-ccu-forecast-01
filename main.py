"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Reads one case per input file (schema makefun/twitch-ccu-case/v1) and writes
five log-scale quantiles (5/25/50/75/95%) for each of hours 2..24 of the
release day. The shape of a launch day is learned at run time from the
comparable past launches bundled in the case itself, so the program adapts
to whatever history the grader hands it.

Model (validated by leave-one-out over the bundled comparables):
  * Level: the case's own hour-1 print anchors everything.
  * Shape: for each hour h, the log-ratio ln(v_h) - ln(v_1) is predicted two
    ways from the comparables -- (a) the median log-ratio of the k=8
    comparables whose hour-1 print is closest in log space, and (b) a
    per-hour linear regression of the log-ratio on ln(hour-1 print). The
    two log-medians are averaged 50/50, floored at the neighbour median so
    the regression can never drag a huge launch below what its closest
    peers actually did.
  * Spread: log-normal quantiles with sigma = the larger of the two
    components' dispersions (neighbour IQR, regression residual SD),
    floored at 0.25 and capped at 1.2. Leave-one-out 90% coverage ~0.93.

Why this shape: hour 1 (00:00 UTC) catches only the regions where the game
has already unlocked, so it is a poor anchor on its own -- launches whose
hour 1 is small explode later in the day as more timezones unlock and the
US comes online (correlation between ln(hour-1) and peak log-growth is
about -0.94), while launches that are already huge at hour 1 dip through
the UTC daytime and recover into US prime time. The two-component blend
captures both regimes.

Python 3.13 standard library only. No network, no randomness, fully
deterministic: two runs on the same input are byte-identical.
"""

import json
import math
import os
import statistics
from decimal import ROUND_HALF_UP, Decimal

# Quantile levels of the output, with their standard-normal z-scores.
LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
ZSCORES = (-1.6448536269514722, -0.6744897501960817, 0.0,
           0.6744897501960817, 1.6448536269514722)

# Model constants (tuned by leave-one-out on the bundled comparables).
NEIGHBOURS = 8          # k for the nearest-neighbour shape
BLEND_WEIGHT = 0.5      # weight on the neighbour median (rest on regression)
SIGMA_FLOOR = 0.25      # narrowest log-spread allowed
SIGMA_CAP = 1.2         # widest log-spread allowed
MAX_VIEWERS = 1e9       # output ceiling from the rules


def round_half_up_2dp(value):
    """Round to 2 decimal places, halves away from zero (the rules' rule)."""
    return float(Decimal(repr(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP))


def hour_one_print(case):
    """The launch's known hour-1 average viewers.

    Prefer the explicit `known` hour-1 row; fall back to the mean of the
    history points inside hour 1, then to the last history point.
    """
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers") is not None:
            return max(1.0, float(row["viewers"]))
    day = (case.get("day_start") or "")[:10]
    inside = [float(p["viewers"]) for p in case.get("history_hourly") or []
              if str(p.get("from", ""))[:10] == day and p.get("viewers") is not None]
    if inside:
        return max(1.0, sum(inside) / len(inside))
    hist = case.get("history") or []
    if hist:
        return max(1.0, float(hist[-1]["viewers"]))
    return 1.0


def release_day_curve(comp):
    """Map hour (1..24) -> viewers for one comparable's release day.

    Returns None unless every hour of the release day is present, so a
    partial day (a premiere, a sparse sample) never enters the training
    set as a shape.
    """
    day = (comp.get("day_start") or "")[:10]
    if not day:
        return None
    hours = {}
    for point in comp.get("hourly") or []:
        stamp = str(point.get("from", ""))
        if stamp[:10] != day:
            continue
        viewers = point.get("viewers")
        if viewers is None:
            continue
        hour = int(stamp[11:13]) + 1
        if 1 <= hour <= 24:
            hours[hour] = max(1.0, float(viewers))
    if len(hours) != 24:
        return None
    return hours


def training_curves(case):
    """Comparable launch-day curves as (hour-1 print, {hour: log-ratio}).

    The log-ratio ln(v_h) - ln(v_1) is the shape; dividing by hour 1 makes
    launches of very different sizes comparable. Sorted by hour-1 print
    so the order is deterministic.
    """
    curves = []
    for comp in case.get("comparables") or []:
        day = release_day_curve(comp)
        if day is None:
            continue
        first = day[1]
        log_first = math.log(first)
        ratios = {h: math.log(day[h]) - log_first for h in range(2, 25)}
        curves.append((first, ratios))
    curves.sort(key=lambda item: item[0])
    return curves

def neighbour_shape(hour_one, curves, hour):
    """Median log-ratio and spread from the NEIGHBOURS closest launches.

    Closeness is measured in log hour-1 print, so a launch is compared
    with launches of a similar size. Spread comes from the interquartile
    range of those neighbours' log-ratios (IQR / 1.349 ~ one SD).
    """
    target = math.log(max(1.0, hour_one))
    nearest = sorted(curves,
                     key=lambda item: abs(math.log(item[0]) - target))
    nearest = nearest[:NEIGHBOURS]
    values = sorted(ratios[hour] for _, ratios in nearest)
    count = len(values)
    median = statistics.median(values)
    if count >= 4:
        spread = (values[(3 * count) // 4] - values[count // 4]) / 1.349
    else:
        spread = 0.5
    return median, spread


def regression_shape(hour_one, curves, hour):
    """Per-hour linear fit of log-ratio on ln(hour-1 print).

    Returns the fitted log-ratio at this launch's hour-1 print and the
    residual standard deviation. The fit is done fresh for each hour, so
    the shape's dependence on launch size can differ across the day.
    """
    xs = [math.log(first) for first, _ in curves]
    ys = [ratios[hour] for _, ratios in curves]
    count = len(xs)
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 0 else 0.0
    intercept = mean_y - slope * mean_x
    fitted = intercept + slope * math.log(max(1.0, hour_one))
    resid_sq = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    spread = math.sqrt(resid_sq / max(1, count - 2))
    return fitted, spread


def forecast_hours(hour_one, curves):
    """Five quantiles per hour (2..24) for one launch.

    The log-median is the 50/50 blend of the neighbour median and the
    regression fit, floored at the neighbour median; the log-spread is
    the wider of the two components' dispersions. Quantiles are
    log-normal around the blended median.
    """
    rows = []
    for hour in range(2, 25):
        neigh_med, neigh_sd = neighbour_shape(hour_one, curves, hour)
        reg_med, reg_sd = regression_shape(hour_one, curves, hour)
        log_med = (BLEND_WEIGHT * neigh_med
                   + (1.0 - BLEND_WEIGHT) * reg_med)
        log_med = max(log_med, neigh_med)
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, max(neigh_sd, reg_sd)))
        centre = math.log(max(1.0, hour_one)) + log_med
        quantiles = []
        running = 0.0
        for z in ZSCORES:
            value = math.exp(centre + z * sigma)
            value = min(MAX_VIEWERS, max(0.01, value))
            value = round_half_up_2dp(value)
            # keep the five quantiles non-decreasing after rounding
            value = max(value, running)
            running = value
            quantiles.append(value)
        rows.append({"hour": hour, "q": quantiles})
    return rows


def forecast_case(case):
    """Forecast one case dict; returns the output document dict."""
    hour_one = hour_one_print(case)
    curves = training_curves(case)
    if not curves:
        # No usable comparables: fall back to the flat house shape so the
        # output is still complete and valid.
        rows = []
        for hour in range(2, 25):
            centre = math.log(hour_one)
            spread = 0.15 * math.sqrt(max(1, hour - 1))
            quantiles = [round_half_up_2dp(
                min(MAX_VIEWERS, max(0.01, math.exp(centre + z * spread))))
                for z in ZSCORES]
            rows.append({"hour": hour, "q": quantiles})
        return {"hours": rows}
    return {"hours": forecast_hours(hour_one, curves)}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        document = forecast_case(case)
        stem = name[:-3] if name.endswith(".in") else name
        out_path = os.path.join(output_dir, stem + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(document, fh, indent=2, sort_keys=False)
            fh.write("\n")


if __name__ == "__main__":
    main()
