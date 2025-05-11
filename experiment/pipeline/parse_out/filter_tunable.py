#!/usr/bin/env python

# Recursively unpack a .yaml file and do structural pattern matching.
import sys
import re

import yaml


# Match on YAML structure:
# KEY:
#   SUBKEY: PATTERN
KEY = 'variational_method'
SUBKEY = 'value'
PATTERN = '(Deterministic*|Meanfield)'


def match_key_value(obj, key, pattern):
    match obj:
        case dict() if key in obj:
            return match_key_value(obj.get(key), SUBKEY, pattern)
        case dict():
            return any(match_key_value(v, key, pattern) for v in obj.values())
        case list():
            return any(match_key_value(v, key, pattern) for v in obj)
        case str():
            return re.search(pattern, obj) is not None

    return False


def main():
    """Take in CLI stream of YAML files, filter out invalid files"""

    files = sys.stdin.read().splitlines(keepends=False)

    for name in files:

        with open(name, 'r') as file:
            data = yaml.safe_load(file)

            if match_key_value(data, KEY, PATTERN):
                print(name)


if __name__ == "__main__":
    main()
