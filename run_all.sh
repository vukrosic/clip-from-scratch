#!/bin/sh
# Runs every step in order. About 45 minutes on a laptop CPU.
set -e
cd "$(dirname "$0")"
for f in step1_world step2_fixed_list step3_two_networks step4_pull step5_never_shown step6_what_matters step7_word_order exercise tests; do
  echo "== $f =="
  python3 $f.py
done
