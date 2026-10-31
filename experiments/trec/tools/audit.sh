# Deterministic anti-autoresearch pass on tmp/trec_audit (run on a vis node). It first copies the site's page and
# extracts its English text (page_text.py, long captions included), so the audit never reads a stale copy (2026-10-31:
# three runs that day audited the Oct 30 text because page_text.py had been run in the wrong directory).
SITE=${SITE:-/scratch/gpfs/GROUP/USER/tmp/trec_site_wt}; TOOLS=$(dirname "$(readlink -f "$0")")
cp "$SITE/exps/trec.html" /scratch/gpfs/GROUP/USER/tmp/trec_audit/trec.html || exit 1
(cd /scratch/gpfs/GROUP/USER/tmp/trec_audit && nice -n 19 python3 "$TOOLS/page_text.py") || exit 1
export CLAUDE_PLUGIN_ROOT=/home/USER/.claude/plugins/cache/wenhaochai/anti-autoresearch/0.1.7; ROOT="${CLAUDE_PLUGIN_ROOT}/support"; PAPER_DIR=/scratch/gpfs/GROUP/USER/tmp/trec_audit; SLUG=trec-page
nice -n 19 python3 "$ROOT/tools/build_manifest.py" --paper-id "$SLUG" --dir "$PAPER_DIR" --pdf-text "$PAPER_DIR/paper.txt" --out "$PAPER_DIR/artifact_manifest.json" >/dev/null || exit 1
L=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[\"observability_level\"])" "$PAPER_DIR/artifact_manifest.json")
nice -n 19 python3 "$ROOT/tools/build_claim_ledger.py" --paper-id "$SLUG" --pdf-text "$PAPER_DIR/paper.txt" --observability-level "$L" --out "$PAPER_DIR/claims.json"
nice -n 19 python3 "$ROOT/tools/check_numeric_consistency.py" --ledger "$PAPER_DIR/claims.json" --out "$PAPER_DIR/consistency-audit.deterministic.findings.json"
nice -n 19 python3 "$ROOT/tools/check_presentation.py" --ledger "$PAPER_DIR/claims.json" --out "$PAPER_DIR/presentation-signals.deterministic.findings.json"
nice -n 19 python3 "$ROOT/tools/adjudicate_findings.py" --findings "$PAPER_DIR"/*.deterministic.findings.json --ledger "$PAPER_DIR/claims.json" --paper-id "$SLUG" --observability-level "$L" --taxonomy-version 0.5 --limitation "Deterministic-only run (no cross-model reviewer in this session): semantic + code-level dimensions were NOT run." --limitation "Source is the English text of an HTML page (paper.txt); charts are drawn from assets/data JSON and were not read by the tools." --out "$PAPER_DIR/report.json" --md "$PAPER_DIR/REPORT.md"
