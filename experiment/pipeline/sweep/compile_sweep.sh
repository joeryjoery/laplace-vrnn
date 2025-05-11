#!/usr/bin/env bash

# Parse Console arguments
pass_through=""
while [ "$#" -gt 0 ]; do
  case $1 in
    -C|--config)
      # Capture variable length arguments to -C/ --config
      shift
      option_config=()
      while [ "$#" -gt 0 ] && [[ $1 != -* ]]; do
        option_config+=("$1")
        shift
      done
      ;;

    -P|--problem) option_problem="$2"; shift 2;;
    -E|--entity) option_entity="$2"; shift 2;;
    -N|--name) option_name="$2"; shift 2;;
    -O|--out) option_out="$2"; shift 2;;

    *) pass_through="$pass_through $1"; shift;;
  esac
done

# Throw away parsed CLI arguments
shift "$((OPTIND-1))"

# Check if Problem-Config is specified and add parsed configs to option_config
if [ -n "${option_problem+x}" ]; then

  # Format config database
  problem_configs=$(<"$option_problem")
  problem_configs=${problem_configs//$'\n'/ }
  problem_configs=${problem_configs//$'\r'/ }

  read -ra problem_elements <<< "$problem_configs"
  option_config+=("${problem_elements[*]}")
fi

# Check strictly required arguments
if [[ -z "${option_name+x}" ]]; then
  echo "Name is Undefined. Specify an argument with -n NAME" >&2
  exit 1;
fi

# Join config-array to a single string for easier splitting.
config_str="${option_config[*]}"

# Generate a sweep yaml conditional on extra arguments
result=$(
python pipeline/sweep/format_config.py \
  -M src \
  -C $config_str \
  -O "${option_out:-$HOME/out}" \
  --token $option_name \
  --wandb $option_name \
  --entity $option_entity \
  $pass_through
)

cleanup_tmps() {
  rm -f "$tmp_config"
  rm -f "$tmp_sweep_out"
}
# Ensure that temporary files are always cleaned
trap cleanup_tmps EXIT ERR

# Create temporary file to store yaml data to be used for wandb sweep
tmp_config=$(mktemp)
tmp_sweep_out=$(mktemp)
echo "$result" > "$tmp_config"

# Write the sweep config to the specified directory (separate from run-data!)
out_dir="${option_out:-$HOME/out}/$option_name"
mkdir -p $out_dir
export WANDB_DIR="$out_dir"

# Create sweep.
wandb sweep \
  -p $option_name \
  --program axme-run \
  --entity $option_entity \
  --name $option_name \
  $tmp_config 2>&1 | tee $tmp_sweep_out

out=$(<$tmp_sweep_out)


# Undo Export
unset WANDB_DIR

# Extract the sweep ID using awk and sed
sweep_id=$(
  echo "$out" | \
  sed 's/\x1B\[[0-9;]*[JKmsu]//g' | \
  awk -F': ' '/wandb: Creating sweep with ID:/ {print $NF}' | \
  awk '{gsub(/[ \r\n]+$/, ""); print}'  # Removes \n \r
)

if [ -z "$sweep_id" ]; then
  echo "Something went wrong when calling 'wandb sweep'. Output Log: " >&2
  echo "$out" >&2
  echo "Exiting" >&2
  exit 1;
fi

# Generate runable script to execute the wandb agent.
runable="pipeline/sweep/run/$sweep_id.sh"

cat > $runable <<EOL
#!/usr/bin/env bash
# Call this script as $runable NUM_AGENTS CHUNK_SIZE

# Set output directory for the agent
export WANDB_DIR="$out_dir"
export AXME_OUT_PREFIX="$out_dir"

num_agents="\${1:-1}"
chunk_size="\${2:-1}"

# Create 'num_agents' independent wandb-agents to perform max. 'chunk_size' runs.
for _ in \$(seq 1 "\$num_agents"); do
  wandb agent -p $option_name --entity $option_entity --count "\$chunk_size" $sweep_id  # &
  sleep 1
done

# Undo Export
unset WANDB_DIR
unset AXME_OUT_PREFIX
EOL

# Make generated script executable and communicate to the user how to run it.
chmod +x "$runable"
echo "Generated Agent Script -- Sweep ID: $sweep_id"
echo "Run the wandb Agent with:"
echo "./run.sh $runable \$NUM_AGENTS \$CHUNK_SIZE"

