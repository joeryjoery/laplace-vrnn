#!/usr/bin/env python
""" Group a list of config files based on matching fields."""
from __future__ import annotations
import sys
import yaml

SEP: str = '/'
DESC_KEY: str = 'desc'
EXCLUDE_BUCKET_FIELDS: list[str] = [s + SEP for s in ['rng', 'evaluator']]
INCLUDE_UNIQUE_FIELDS: list[str] = [s + SEP for s in ['rng', 'model']]


def flatten_dict(d: dict, separator: str = '_', parent_key: str = '') -> dict:
    """Flatten arbitrarily nested dictionaries with a key separator. """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, separator, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def filter_descriptions(my_dict: dict, field: str) -> dict:
    """Remove the given field from an arbitrarily nested dictionary. """

    # Create copy without DESC_KEY
    my_dict = {k: v for k, v in my_dict.items() if k != field}

    # Filter sub-dictionaries on DESC_KEY
    my_dict = {
        k: (filter_descriptions(v, field) if isinstance(v, dict) else v)
        for k, v in my_dict.items()
    }

    return my_dict


def are_dicts_equal(a: dict, b: dict) -> bool:
    """Check if two dictionaries are the same. """
    if isinstance(a, dict) and isinstance(b, dict):
        if len(a) != len(b):
            return False
        for key, value in b.items():
            if key not in b or not are_dicts_equal(value, b[key]):
                return False
        return True
    else:
        return a == b


def bucket_dicts(
        *dicts: tuple[str, dict],
        exclude: list[str] | None = None
) -> list[dict]:
    """Group a list of dictionaries together based on their values. """

    buckets = dict()
    grouped_configs = dict()

    for i, (name, config) in enumerate(dicts):

        flat = flatten_dict(config, separator=SEP)

        filtered = flat
        if exclude:
            filtered = {
                k: v for k, v in flat.items()
                if not any(n in k for n in exclude)
            }

        idx = hash(str(filtered))
        if idx not in buckets:
            buckets[idx], grouped_configs[idx] = list(), list()
        else:
            # Extract 1 reference config
            reference = grouped_configs[idx][0]

            if not are_dicts_equal(filtered, reference):
                # Not sponsored by Supercell
                raise RuntimeError(
                    f"Clash of Hash:\nA: {filtered}\nB: {reference}"
                )

        buckets[idx].append(i)
        grouped_configs[idx].append(filtered)

    return [
        {dicts[i][0]: dicts[i][1] for i in v}
        for k, v in buckets.items()
    ]


def filter_unique_values(
        my_dict: dict,
        on: list[str] | None = None
) -> dict:
    """Return a new dictionary with strictly unique values. """
    flattened_values = {
        k: flatten_dict(v, separator=SEP)
        for k, v in my_dict.items()
    }

    filtered = flattened_values
    if on:
        # Too much comprehension nesting, but lazy...
        filtered = {
            k: {
                sk: sv for sk, sv in v.items()
                if any(n in sk for n in on)
            }
            for k, v in flattened_values.items()
        }

    seen, keys = set(), list()
    for k, v in filtered.items():
        idx = hash(str(v))

        if idx not in seen:
            seen.add(idx)
            keys.append(k)

    return {k: my_dict[k] for k in keys}


if __name__ == "__main__":

    # Receive candidate files from console stream
    files = sys.stdin.read().strip().splitlines(keepends=False)

    # Read in config data and remove annotations.
    configs = {f: yaml.safe_load(open(f)) for f in files}
    filtered_configs = filter_descriptions(configs, DESC_KEY)

    # Group configs together based on field values.
    split = [(k, v) for k, v in filtered_configs.items()]
    grouped = bucket_dicts(*split, exclude=EXCLUDE_BUCKET_FIELDS)

    # Throw away duplicate runs.
    unique = [
        filter_unique_values(group, on=INCLUDE_UNIQUE_FIELDS)
        for group in grouped
    ]

    # Empty results into console stream
    for group in unique:
        print(*group.keys())
