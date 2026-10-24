#!/bin/bash
# cbw_rate.sh LOG [MIN_IT] [MAX_IT]: seconds per train step from the tqdm "Progress on:train" lines of a run log (one line
# a minute), between the first line at >= MIN_IT steps (default 20: after compile) and the last line at <= MAX_IT
# (default: no limit). Prints "steps <a>-><b> in <s> s: <s/it> s/it". A count like 5.38kit is rounded to 10 steps, so
# use spans of a few hundred steps. Eval and checkpoint pauses inside the span count as step time.
log=$1; lo=${2:-20}; hi=${3:-1000000000}
tr '\r' '\n' < "$log" | grep -a "Progress on:train [0-9]" | awk -v lo="$lo" -v hi="$hi" '
  { split($2, t, ":"); s = t[1] * 3600 + t[2] * 60 + t[3]
    if (s < prev) day += 86400; prev = s; s += day
    m = $0; sub(/.*Progress on:train /, "", m); sub(/\/.*/, "", m)
    k = (m ~ /kit$/) ? 1000 : 1; sub(/k?it$/, "", m); n = m * k
    if (n >= lo && n <= hi) { if (!have) { a = n; ta = s; have = 1 } b = n; tb = s } }
  END { if (have && b > a) printf "steps %d->%d in %d s: %.3f s/it\n", a, b, tb - ta, (tb - ta) / (b - a); else print "no rate (too few progress lines)" }'
