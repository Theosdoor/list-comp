#!/bin/bash
#SBATCH --job-name=pdf2md
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=5:00:00
#SBATCH --mem=16G

# 1. Activate Environment
cd /home2/nchw73/Year4/L4_Project/list-comp-priv/
uv sync
uv add marker-pdf
source .venv/bin/activate

echo "Job running on node: $(hostname)"
echo "------------------------------------------------------"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
echo "------------------------------------------------------"

# 2. Convert PDFs
python3 - <<'EOF'
from pathlib import Path
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

models = create_model_dict()
converter = PdfConverter(artifact_dict=models)

input_dir = Path("reference")

for pdf in input_dir.glob("*.pdf"):
    print(f"Converting: {pdf.name}")
    rendered = converter(str(pdf))
    out_path = input_dir / (pdf.stem + ".md")
    out_path.write_text(rendered.markdown)
    print(f"  -> {out_path}")

print("Done.")
EOF

uv remove marker-pdf