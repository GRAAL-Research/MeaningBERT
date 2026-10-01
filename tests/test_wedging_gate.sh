# Verifie la porte, sans GPU : on simule nvidia-smi.
set -u
tmp=$(mktemp -d); mkdir -p "$tmp/bin"
cat > "$tmp/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  *name*) echo "$FAKE_GPU" ;;
  *compute_cap*) echo "6.1" ;;
esac
EOF
chmod +x "$tmp/bin/nvidia-smi"
export PATH="$tmp/bin:$PATH"

essai() {  # nom, carte simulee, cellules, motif attendu
  FAKE_GPU="$2" CELLS="$3" REPO="$tmp" VENV="$tmp" CORPUS="$tmp" ROOT="$tmp/res" MIN_FREE_GB=0 \
    bash src/training/run_polarity.sh test 2>&1 | grep -q "$4" \
    && echo "  OK   $1" || echo "  ECHEC $1 (attendu : $4)"
}
essai "bert refuse sur une 1080 Ti"            "NVIDIA GeForce GTX 1080 Ti" "bert:42"               "fige une"
essai "stsb refuse sur une 1080 Ti"            "NVIDIA GeForce GTX 1080 Ti" "stsb-roberta-base:42"  "fige une"
essai "bert accepte sur une P5000"             "Quadro P5000"               "bert:42"               "RUN"
essai "deberta accepte sur une 1080 Ti"        "NVIDIA GeForce GTX 1080 Ti" "deberta-v3-base:42"    "RUN"
essai "stsb accepte sur un TITAN Xp"           "NVIDIA TITAN Xp"            "stsb-roberta-base:42"  "RUN"
rm -rf "$tmp"
