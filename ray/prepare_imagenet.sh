#!/bin/bash
#===============================================================================
# Download ImageNet-1K (ILSVRC2012) from HuggingFace
#
# Downloads ~138 GB to /nrs/ml_datasets/imagenet
# Requires HuggingFace token with access to ILSVRC/imagenet-1k
#
# Before running:
#   1. Create a HuggingFace account at https://huggingface.co
#   2. Accept the license at https://huggingface.co/datasets/ILSVRC/imagenet-1k
#   3. Create a token at https://huggingface.co/settings/tokens
#   4. Export it:  export HF_TOKEN="hf_..."
#
# Usage:
#   ./prepare_imagenet.sh [--venv=PATH] [--data-dir=PATH]
#
# Defaults:
#   venv:     ~/ray_env
#   data-dir: /nrs/ml_datasets/imagenet
#===============================================================================

VENV_PATH="${HOME}/ray_env"
DATA_DIR="/nrs/ml_datasets/imagenet"

while [[ $# -gt 0 ]]; do
    case $1 in
        --venv=*)     VENV_PATH="${1#*=}"; shift ;;
        --data-dir=*) DATA_DIR="${1#*=}"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo ""
    echo "ImageNet-1K is a gated dataset. To download it:"
    echo "  1. Create a HuggingFace account at https://huggingface.co"
    echo "  2. Accept the license at https://huggingface.co/datasets/ILSVRC/imagenet-1k"
    echo "  3. Create a token at https://huggingface.co/settings/tokens"
    echo "  4. Run:  export HF_TOKEN=\"hf_...\""
    exit 1
fi

echo "================================================================"
echo "ImageNet-1K (ILSVRC2012) Download"
echo "================================================================"
echo "Venv:     $VENV_PATH"
echo "Data dir: $DATA_DIR"
echo "Size:     ~138 GB"
echo "================================================================"

# LSF does not create the log directory; without it the job output is lost
mkdir -p ../output

# HF_TOKEN is NOT written into the job script (LSF stores it and shows it in
# bjobs -l); bsub forwards the submission environment to the job instead.
export HF_TOKEN

cat << EOF | bsub
#!/bin/bash
#BSUB -J imagenet_download
#BSUB -n 8
#BSUB -o ../output/imagenet_download_%J.out
#BSUB -e ../output/imagenet_download_%J.err
#BSUB -W 24:00

source ${VENV_PATH}/bin/activate
if [ -z "\$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN did not reach the job environment"
    exit 1
fi

mkdir -p ${DATA_DIR}

echo "Downloading ImageNet-1K (ILSVRC2012) from HuggingFace..."
echo "Target: ${DATA_DIR}"
echo ""

python3 - << 'PYEOF'
import os
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

from huggingface_hub import snapshot_download

data_dir = "${DATA_DIR}"
print(f"Downloading to: {data_dir}")
print("This will take a while (~138 GB)...")

snapshot_download(
    repo_id="ILSVRC/imagenet-1k",
    repo_type="dataset",
    local_dir=data_dir,
)

print()
print("Download complete.")
for f in sorted(os.listdir(data_dir)):
    path = os.path.join(data_dir, f)
    if os.path.isfile(path):
        size = os.path.getsize(path) / 1e9
        print(f"  {f}: {size:.1f} GB")
PYEOF
if [ \$? -ne 0 ]; then
    echo "ERROR: ImageNet download failed"
    exit 1
fi

echo ""
echo "ImageNet preparation complete."
ls -lh ${DATA_DIR}/
EOF
