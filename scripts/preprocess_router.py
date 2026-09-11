"""Routing logic for conditional preprocessing: (weather, timeofday) -> filter pipeline.

Decision table (weather corrected before timeofday, always):

              daytime        night                          dawn/dusk
clear         raw            clahe_bilateral                raw
rainy         derain         derain -> clahe_bilateral       derain
foggy         dcp_dehaze     clahe_bilateral (DCP skipped)   dcp_dehaze
overcast      histeq_lab     clahe_bilateral                histeq_lab
snowy         raw            clahe_bilateral                raw
partly cloudy raw            clahe_bilateral                raw
undefined     raw            raw                            raw

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
        "night": [filt.clahe_bilateral],
        "dawn/dusk": [],
    },
    "rainy": {
        "daytime": [filt.derain],
        "night": [filt.derain, filt.clahe_bilateral],
        "dawn/dusk": [filt.derain],
    },
    "foggy": {
        "daytime": [filt.dcp_dehaze],
        "night": [filt.clahe_bilateral],
        "dawn/dusk": [filt.dcp_dehaze],
    },
    "overcast": {
        "daytime": [filt.histeq_lab],
        "night": [filt.clahe_bilateral],
        "dawn/dusk": [filt.histeq_lab],
    },
    "snowy": {
        "daytime": [],
        "night": [filt.clahe_bilateral],
        "dawn/dusk": [],
    },
    "partly cloudy": {
        "daytime": [],
        "night": [filt.clahe_bilateral],
        "dawn/dusk": [],
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
