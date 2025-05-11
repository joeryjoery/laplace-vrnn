#!/usr/bin/env bash

tunable_runs=$(./pipeline/parse_out/extract_tunables.sh $1)

shift

# Parse Console arguments
pass_through=""
while [ "$#" -gt 0 ]; do
  case $1 in
    -S|--sweep)
      # Capture variable length arguments to -C/ --config
      shift
      option_sweep=()
      while [ "$#" -gt 0 ] && [[ $1 != -* ]]; do
        option_sweep+=("$1")
        shift
      done
      ;;
    -C|--config)
      # Capture variable length arguments to -C/ --config
      shift
      option_config=()
      while [ "$#" -gt 0 ] && [[ $1 != -* ]]; do
        option_config+=("$1")
        shift
      done
      ;;
    *) pass_through="$pass_through $1"; shift;;
  esac
done

cleanup_tmps() {
  rm -f "$tmp_config"
  rm -f "$tmp_format"
}
# Ensure that temporary files are always cleaned
trap cleanup_tmps EXIT ERR

while read -r group; do
    reference=$(echo "$group" | cut -d' ' -f1)

    tmp_config=$(mktemp)
    tmp_format=$(mktemp)
    echo "$group" | python pipeline/parse_out/tunable_to_sweep_config.py > "$tmp_config"

    pipeline/sweep/compile_sweep.sh \
      $pass_through \
      -S $tmp_config ${option_sweep[*]} \
      -C $reference ${option_config[*]} 2>&1 | tee $tmp_format

    rm -f "$tmp_config"
done <<< "$tunable_runs"
