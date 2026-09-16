"""Routing logic for conditional preprocessing: (weather, timeofday) -> filter pipeline.

Decision table (weather corrected before timeofday, always):

              daytime           night                           dawn/dusk
clear         raw               adaptive_enhance                adaptive_enhance
rainy         derain            derain -> adaptive_enhance      derain
foggy         dcp_dehaze        adaptive_enhance (DCP skipped)  dcp_dehaze
overcast      adaptive_enhance  adaptive_enhance                adaptive_enhance
snowy         raw               adaptive_enhance                adaptive_enhance
partly cloudy raw               adaptive_enhance                adaptive_enhance
undefined     raw               raw                             raw

dawn/dusk's "raw" cells (clear/snowy/partly cloudy) are swapped for
adaptive_enhance - twilight ambient light is dim, and those buckets were
previously routed like daytime (no brightening at all), which is why
dawn/dusk output looked just as dark as raw input.

overcast used histeq_lab (global histogram equalization) before. Measured
inside vehicle boxes (YOLO labels, n=100 daytime / n=80 dawn-dusk), global
equalization crushes vehicle pixels to black - the sky dominates the
histogram, so the mapping steepens over the sky's range and compresses the
midtones where vehicles actually live:

  overcast/daytime    dark% inside boxes: raw 0.02 -> histeq 1.28 (64x)
  overcast/dawn-dusk  dark% inside boxes: raw 0.03 -> histeq 1.69 (56x)

adaptive_enhance beats it on every metric in both buckets (brighter, higher
contrast, more gradient energy) while destroying nothing - it even clips
fewer highlights than the raw input, because its gamma pass lifts darks
before CLAHE instead of redistributing them, and the bilateral pass cleans up
what CLAHE amplifies.

The raw cells (clear/snowy/partly cloudy daytime) stay raw deliberately:
measured inside vehicle boxes, those buckets already match or beat
clear/daytime on exposure, so there is no defect to correct there. Snow's
highlight clipping sits in sky/snowbanks, outside every box - and snow breaks
the Dark Channel Prior assumption, so forcing dcp_dehaze on it crushes
vehicles (dark% 6.6 -> 41.9).

To add a filter or change a bucket's pipeline, edit ROUTING_TABLE only.
"""

import preprocessing_filters as filt

TIMEOFDAY_ALIASES = {
    "daytime": "daytime",
    "day": "daytime",
    "night": "night",
    "dawn/dusk": "dawn/dusk",
    "dawn": "dawn/dusk",
    "dusk": "dawn/dusk",
}

# weather -> {timeofday_bucket: [filters in apply order]}
ROUTING_TABLE = {
    "clear": {
        "daytime": [],
        "night": [filt.adaptive_enhance],
        "dawn/dusk": [filt.adaptive_enhance],
    },
    "rainy": {
        "daytime": [filt.derain],
        "night": [filt.derain, filt.adaptive_enhance],
        "dawn/dusk": [filt.derain],
    },
    "foggy": {
        "daytime": [filt.dcp_dehaze],
        "night": [filt.adaptive_enhance],
        "dawn/dusk": [filt.dcp_dehaze],
    },
    "overcast": {
        "daytime": [filt.adaptive_enhance],
        "night": [filt.adaptive_enhance],
        "dawn/dusk": [filt.adaptive_enhance],
    },
    "snowy": {
        "daytime": [],
        "night": [filt.adaptive_enhance],
        "dawn/dusk": [filt.adaptive_enhance],
    },
    "partly cloudy": {
        "daytime": [],
        "night": [filt.adaptive_enhance],
        "dawn/dusk": [filt.adaptive_enhance],
    },
}


def normalize_weather(weather):
    if weather is None:
        return None
    weather = weather.strip().lower()
    return weather if weather in ROUTING_TABLE else None


def normalize_timeofday(timeofday):
    if timeofday is None:
        return None
    return TIMEOFDAY_ALIASES.get(timeofday.strip().lower())


def get_pipeline(weather, timeofday):
    """Return the ordered list of filter functions for this (weather, timeofday).

    Any missing/undefined/unrecognized attribute falls back to no filters
    (raw copy), matching the decision table's "undefined" row.
    """
    w = normalize_weather(weather)
    tod = normalize_timeofday(timeofday)
    if w is None or tod is None:
        return []
    return ROUTING_TABLE[w][tod]


def bucket_label(weather, timeofday):
    """Human-readable bucket key used purely for counting/logging (not routing).

    Keeps the raw (possibly undefined/missing) values visible in the report
    instead of collapsing them into "undefined" too early.
    """
    w = (weather or "missing").strip().lower() if weather else "missing"
    t = (timeofday or "missing").strip().lower() if timeofday else "missing"
    return f"{w}|{t}"
