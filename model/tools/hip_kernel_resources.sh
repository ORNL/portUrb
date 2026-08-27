#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $0 --executable FILE --output FILE --source-root DIR \\" >&2
  echo "          --roc-obj-ls FILE --llvm-nm FILE --llvm-readobj FILE --llvm-symbolizer FILE" >&2
}

executable=""
output=""
source_root=""
roc_obj_ls=""
llvm_nm=""
llvm_readobj=""
llvm_symbolizer=""

while (( $# > 0 )); do
  case "$1" in
    --executable)      executable="$2";      shift 2 ;;
    --output)          output="$2";          shift 2 ;;
    --source-root)     source_root="$2";     shift 2 ;;
    --roc-obj-ls)      roc_obj_ls="$2";      shift 2 ;;
    --llvm-nm)         llvm_nm="$2";         shift 2 ;;
    --llvm-readobj)    llvm_readobj="$2";    shift 2 ;;
    --llvm-symbolizer) llvm_symbolizer="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$executable" || -z "$output" || -z "$source_root" || -z "$roc_obj_ls" ||
      -z "$llvm_nm" || -z "$llvm_readobj" || -z "$llvm_symbolizer" ]]; then
  usage
  exit 2
fi

if [[ ! -f "$executable" ]]; then
  echo "HIP resource report: executable not found: $executable" >&2
  exit 1
fi

source_root="$(cd "$source_root" && pwd -P)"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/porturb-hip-resources.XXXXXX")"
trap 'rm -rf "$temporary_dir"' EXIT

all_rows="$temporary_dir/all_rows.tsv"
: > "$all_rows"
declare -A seen_images
image_index=0

while IFS= read -r listing; do
  [[ "$listing" == *amdgcn-amd-amdhsa* ]] || continue
  if [[ ! "$listing" =~ offset=([^\&]+)\&size=([^[:space:]]+) ]]; then
    continue
  fi
  offset="${BASH_REMATCH[1]}"
  size="${BASH_REMATCH[2]}"
  image="$temporary_dir/image-${image_index}.co"
  image_index=$((image_index+1))
  dd if="$executable" of="$image" iflag=skip_bytes,count_bytes \
     skip="$offset" count="$size" status=none

  image_key="$(cksum < "$image")"
  if [[ -n "${seen_images[$image_key]:-}" ]]; then
    continue
  fi
  seen_images[$image_key]=1

  metadata_raw="$temporary_dir/metadata-${image_index}.txt"
  metadata="$temporary_dir/metadata-${image_index}.tsv"
  symbols="$temporary_dir/symbols-${image_index}.txt"
  kernels="$temporary_dir/kernels-${image_index}.tsv"
  samples="$temporary_dir/samples-${image_index}.tsv"
  addresses="$temporary_dir/addresses-${image_index}.txt"
  symbolized="$temporary_dir/symbolized-${image_index}.txt"
  locations="$temporary_dir/locations-${image_index}.tsv"
  image_rows="$temporary_dir/rows-${image_index}.tsv"

  "$llvm_readobj" --notes "$image" > "$metadata_raw"
  awk '
    function trim(value) {
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      return value
    }
    function value_after_colon(value) {
      sub(/^[^:]*:/, "", value)
      return trim(value)
    }
    function emit() {
      if (name != "") print name "\t" vgpr "\t" vgpr_spills "\t" sgpr_spills "\t" private_segment
    }
    /^  - \.agpr_count:/ {
      emit()
      name = ""; vgpr = 0; vgpr_spills = 0; sgpr_spills = 0; private_segment = 0
      next
    }
    /^[[:space:]]+\.name:/                       { name = value_after_colon($0); next }
    /^[[:space:]]+\.vgpr_count:/                 { vgpr = value_after_colon($0); next }
    /^[[:space:]]+\.vgpr_spill_count:/           { vgpr_spills = value_after_colon($0); next }
    /^[[:space:]]+\.sgpr_spill_count:/           { sgpr_spills = value_after_colon($0); next }
    /^[[:space:]]+\.private_segment_fixed_size:/ { private_segment = value_after_colon($0); next }
    END { emit() }
  ' "$metadata_raw" > "$metadata"

  "$llvm_nm" --defined-only --format=posix "$image" > "$symbols"
  awk -F '\t' '
    NR == FNR {
      split($0, fields, " ")
      if (fields[2] == "T" || fields[2] == "t") {
        address[fields[1]] = fields[3]
        size[fields[1]] = fields[4]
      }
      next
    }
    $1 in address {
      kernel_count++
      print kernel_count "\t" $1 "\t" address[$1] "\t" size[$1] "\t" $2 "\t" $3 "\t" $4 "\t" $5
    }
  ' "$symbols" "$metadata" > "$kernels"

  : > "$samples"
  : > "$addresses"
  while IFS=$'\t' read -r kernel_id kernel_name address_hex size_hex vgpr vgpr_spills sgpr_spills private_segment; do
    address_decimal=$((16#$address_hex))
    size_decimal=$((16#$size_hex))
    if (( size_decimal <= 4 )); then
      sample_count=1
    else
      sample_count=$((size_decimal/128))
      (( sample_count < 2 )) && sample_count=2
      (( sample_count > 48 )) && sample_count=48
    fi
    for ((sample_index=0; sample_index < sample_count; sample_index++)); do
      if (( sample_count == 1 )); then
        sample_address=$address_decimal
      else
        sample_address=$((address_decimal + sample_index*(size_decimal-4)/(sample_count-1)))
      fi
      printf -v sample_hex '0x%x' "$sample_address"
      printf '%s\t%s\n' "$sample_hex" "$kernel_id" >> "$samples"
      printf '%s\n' "$sample_hex" >> "$addresses"
    done
  done < "$kernels"

  if [[ ! -s "$addresses" ]]; then
    continue
  fi
  "$llvm_symbolizer" --print-address --inlines --demangle --obj="$image" \
    < "$addresses" > "$symbolized"

  awk -F '\t' -v root="$source_root" '
    NR == FNR { sample[$1] = $2; if ($2 > max_id) max_id = $2; next }
    function source_priority(path, exact) {
      if (index(path, root "/") == 1 && path !~ /\/external\/(kokkos|YAKL)\//) return exact ? 4 : 2
      if (index(path, root "/") == 1) return exact ? 3 : 1
      return exact ? 2 : 0
    }
    function record_source(id, path, line, exact, priority) {
      if (path == "" || path == "??" || line <= 0) return
      priority = source_priority(path, exact)
      if (!(id in best_priority) || priority > best_priority[id]) {
        source_file[id] = path
        source_line[id] = line
        best_priority[id] = priority
      }
    }
    /^0x[0-9a-fA-F]+$/ { current = sample[tolower($0)]; next }
    current == "" || $0 == "" { next }
    {
      text = $0
      lambda_start = index(text, "(lambda at ")
      if (lambda_start > 0) {
        location = substr(text, lambda_start+11)
        sub(/\).*/, "", location)
        sub(/:[0-9]+$/, "", location)
        line = location
        sub(/^.*:/, "", line)
        sub(/:[0-9]+$/, "", location)
        record_source(current, location, line+0, 1)
      }

      if (match(text, /launch_parallel_for_untiled<[0-9]+/)) {
        token = substr(text, RSTART, RLENGTH)
        threads = token
        sub(/^.*</, "", threads)
        config[current] = "default/untiled"
        if (threads+0 > 0) config[current] = "Config<" threads ">/untiled"
      } else if (match(text, /launch_parallel_for_tiled<[0-9]+/)) {
        token = substr(text, RSTART, RLENGTH)
        threads = token
        sub(/^.*</, "", threads)
        config[current] = "default/tiled"
        if (threads+0 > 0) config[current] = "Config<" threads ">/tiled"
      }

      if (text ~ /:[0-9]+:[0-9]+$/) {
        location = text
        sub(/:[0-9]+$/, "", location)
        line = location
        sub(/^.*:/, "", line)
        sub(/:[0-9]+$/, "", location)
        record_source(current, location, line+0, 0)
      }
    }
    END {
      for (id=1; id <= max_id; id++) {
        file = source_file[id]
        if (file == "") file = "?"
        if (index(file, root "/") == 1) file = substr(file, length(root)+2)
        line = source_line[id]
        if (line == "") line = "?"
        launch = config[id]
        if (launch == "") launch = "default"
        print id "\t" file "\t" line "\t" launch
      }
    }
  ' "$samples" "$symbolized" > "$locations"

  awk -F '\t' 'BEGIN { OFS="\t" }
    NR == FNR { file[$1]=$2; line[$1]=$3; config[$1]=$4; next }
    { print file[$1], line[$1], config[$1], $5, $6, $7, $8 }
  ' "$locations" "$kernels" > "$image_rows"
  cat "$image_rows" >> "$all_rows"
done < <("$roc_obj_ls" "$executable")

if [[ ! -s "$all_rows" ]]; then
  echo "HIP resource report: no AMD GPU kernels found in $executable" >&2
  exit 1
fi

normalized_rows="$temporary_dir/normalized.tsv"
awk -F '\t' 'BEGIN { OFS="\t" }
  {
    row_count++
    for (field=1; field <= NF; field++) row[row_count,field] = $field
    group = $1 SUBSEP $2
    if ($3 ~ /^Config</) has_configs[group] = 1
  }
  END {
    for (row_index=1; row_index <= row_count; row_index++) {
      group = row[row_index,1] SUBSEP row[row_index,2]
      launch = row[row_index,3]
      if (launch ~ /^default\// && has_configs[group]) sub(/^default/, "Config<0>", launch)
      if (launch == "default/untiled") launch = "default"
      if (has_configs[group]) launch = "autotune " launch
      row[row_index,3] = launch
      print row[row_index,1], row[row_index,2], row[row_index,3], row[row_index,4], row[row_index,5], \
            row[row_index,6], row[row_index,7]
    }
  }
' "$all_rows" | LC_ALL=C sort -t $'\t' -k1,1 -k2,2n -k3,3 > "$normalized_rows"

mkdir -p "$(dirname "$output")"
temporary_output="$(mktemp "${output}.tmp.XXXXXX")"
awk -F '\t' -v executable_name="$(basename "$executable")" '
  function clip(value, width) {
    if (length(value) <= width) return value
    return "..." substr(value, length(value)-(width-4))
  }
  BEGIN {
    print "HIP kernel resources for " executable_name
    print ""
    printf "%-64s | %8s | %-34s | %10s | %12s | %12s | %24s\n", \
           "file", "lineno", "launch_config", "VGPR_count", "VGPR_spills", "SGPR_spills", "private_segment_thread"
    printf "%-64s-+-%8s-+-%-34s-+-%10s-+-%12s-+-%12s-+-%24s\n", \
           "----------------------------------------------------------------", "--------", "----------------------------------", \
           "----------", "------------", "------------", "------------------------"
  }
  {
    printf "%-64s | %8s | %-34s | %10s | %12s | %12s | %24s\n", \
           clip($1,64), $2, clip($3,34), $4, $5, $6, $7
  }
' "$normalized_rows" > "$temporary_output"
mv "$temporary_output" "$output"

row_count="$(wc -l < "$normalized_rows")"
echo "Wrote $row_count HIP kernel rows to $output"
