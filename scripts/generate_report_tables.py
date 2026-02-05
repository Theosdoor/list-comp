"""
Generate report tables and heatmaps for SAEs with special features
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import plotly.io as pio

from src.utils.nb_utils import setup_notebook, load_transformer_model, load_sae
from src.sae.sae_analysis import (
    collect_sae_activations,
    create_feature_heatmaps
)
from src.data.datasets import get_dataset
from torch.utils.data import DataLoader


def parse_sae_comparison():
    """Parse sae_comparison.md to extract SAE info"""
    comparison_file = Path(__file__).parent.parent / "sae_comparison.md"
    
    # Read special features section
    with open(comparison_file, 'r') as f:
        content = f.read()
    
    # Parse special features table
    special_section_start = content.find("## Special Features")
    special_table_start = content.find("| Model |", special_section_start)
    special_table_end = content.find("\n### Top Special Features", special_table_start)
    
    special_lines = content[special_table_start:special_table_end].strip().split('\n')[2:]  # Skip header rows
    
    special_features_data = []
    for line in special_lines:
        if line.strip() and line.startswith('|'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 5:
                model = parts[0]
                n_special = int(parts[1])
                special_pct = parts[2]
                max_corr = float(parts[3])
                mean_abs_corr = float(parts[4])
                special_features_data.append({
                    'model': model,
                    'n_special': n_special,
                    'special_pct': special_pct,
                    'max_corr': max_corr,
                    'mean_abs_corr': mean_abs_corr
                })
    
    # Parse top special features details
    features_section_start = content.find("### Top Special Features by Model")
    features_section = content[features_section_start:]
    
    model_features = defaultdict(list)
    current_model = None
    
    for line in features_section.split('\n'):
        line = line.strip()
        if line.startswith('**sae_'):
            current_model = line.replace('**', '').replace(':', '')
        elif line.startswith('- Feature') and current_model:
            # Parse: - Feature 46: d2_favoring, corr=-0.5601
            parts = line.replace('- Feature ', '').split(':')
            feat_num = int(parts[0])
            feat_info = parts[1].strip().split(',')
            feat_type = feat_info[0].strip()
            corr = float(feat_info[1].replace('corr=', '').strip())
            model_features[current_model].append({
                'feature': feat_num,
                'type': feat_type,
                'corr': corr
            })
    
    # Parse summary table for additional metrics
    summary_section_start = content.find("## Summary Table")
    summary_table_start = content.find("| Model |", summary_section_start)
    summary_table_end = content.find("\n## Best Models", summary_table_start)
    summary_lines = content[summary_table_start:summary_table_end].strip().split('\n')[2:]
    
    summary_data = {}
    for line in summary_lines:
        if line.strip() and line.startswith('|'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 11:
                model = parts[0]
                summary_data[model] = {
                    'd_sae': int(parts[1]),
                    'k': int(parts[2]),
                    'L0': float(parts[3]),
                    'Dead': int(parts[4]),
                    'Dead_pct': parts[5],
                    'Alive': int(parts[6]),
                    'MSE': float(parts[7]),
                    'Exp_Var': float(parts[8]),
                    'Recon_Acc': float(parts[9]),
                    'Acc_Drop': float(parts[10])
                }
    
    return special_features_data, model_features, summary_data


def create_markdown_tables():
    """Create markdown tables organized by k value"""
    special_data, model_features, summary_data = parse_sae_comparison()
    
    # Filter to only models with at least 1 special feature
    models_with_features = [d for d in special_data if d['n_special'] >= 1]
    
    # Group by k value
    k_groups = defaultdict(list)
    for model_info in models_with_features:
        model_name = model_info['model']
        if model_name in summary_data:
            k = summary_data[model_name]['k']
            k_groups[k].append(model_name)
    
    # Generate markdown tables
    tables_md = "# SAE Analysis: Models with Special Features\n\n"
    
    for k in sorted(k_groups.keys()):
        tables_md += f"## Top-K = {k}\n\n"
        
        # Create table header
        tables_md += "| Model | d_sae | L0 | Dead | Alive | MSE | Exp Var | Recon Acc | Acc Drop | N Special | Special % | Max Corr | Features |\n"
        tables_md += "|-------|-------|----|----|-------|-----|---------|-----------|----------|-----------|-----------|----------|---------|\n"
        
        for model_name in k_groups[k]:
            summary = summary_data[model_name]
            special_info = next(d for d in special_data if d['model'] == model_name)
            
            # Get features info
            features = model_features.get(model_name, [])
            features_str = "<br>".join([
                f"F{f['feature']}: {f['type']} ({f['corr']:.4f})"
                for f in features
            ])
            
            tables_md += f"| {model_name} | {summary['d_sae']} | {summary['L0']:.2f} | "
            tables_md += f"{summary['Dead']} | {summary['Alive']} | {summary['MSE']:.4f} | "
            tables_md += f"{summary['Exp_Var']:.4f} | {summary['Recon_Acc']:.4f} | "
            tables_md += f"{summary['Acc_Drop']:.4f} | {special_info['n_special']} | "
            tables_md += f"{special_info['special_pct']} | {special_info['max_corr']:.4f} | "
            tables_md += f"{features_str} |\n"
        
        tables_md += "\n"
    
    return tables_md, models_with_features


def generate_heatmaps(models_list, device="cuda"):
    """Generate feature heatmaps for all models with special features"""
    print(f"Generating heatmaps for {len(models_list)} models...")
    
    # Load base model
    models_dir = str(Path(__file__).parent.parent / "models")
    model, model_config = load_transformer_model('2layer_100dig_64d', device=device, models_dir=models_dir)
    
    # Load validation dataset
    list_len = model_config['list_len']
    n_digits = model_config['n_digits']
    _, val_ds = get_dataset(
        list_len=list_len,
        n_digits=n_digits,
        train_split=0.8,
        no_dupes=False,
        train_dupes_only=False,
        seed=42
    )
    val_dl = DataLoader(val_ds, batch_size=512, shuffle=False)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "demoday_5feb" / "heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for model_info in models_list:
        model_name = model_info['model']
        print(f"\nProcessing {model_name}...")
        
        try:
            # Load SAE
            sae, sae_config = load_sae(
                model_name + '.pt',
                d_model=model_config['d_model'],
                device=device,
                sae_dir=str(Path(__file__).parent.parent / "results" / "sae_models")
            )
            
            # Get activation mean from checkpoint
            sae_path = Path(__file__).parent.parent / "results" / "sae_models" / (model_name + '.pt')
            checkpoint = torch.load(sae_path, map_location=device, weights_only=False)
            act_mean = checkpoint.get("act_mean", torch.zeros(model_config['d_model'])).to(device)
            
            # Collect activations
            d1_all, d2_all, sae_acts_all = collect_sae_activations(
                model, sae, val_dl, act_mean,
                layer_idx=0,
                sep_idx=list_len,
                device=device
            )
            
            # Create heatmaps
            fig = create_feature_heatmaps(
                d1_all, d2_all, sae_acts_all,
                n_digits=n_digits,
                figsize=(20, 20)
            )
            
            # Save as HTML
            output_file = output_dir / f"{model_name}_heatmap.html"
            pio.write_html(fig, str(output_file))
            print(f"Saved: {output_file}")
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue


if __name__ == "__main__":
    # Setup
    device = setup_notebook(seed=42, disable_grad=True)
    
    # Generate tables
    print("Generating tables...")
    tables_md, models_with_features = create_markdown_tables()
    
    # Save tables to report.md
    report_file = Path(__file__).parent.parent / "demoday_5feb" / "report.md"
    with open(report_file, 'w') as f:
        f.write(tables_md)
    print(f"Saved tables to {report_file}")
    
    # Generate heatmaps
    generate_heatmaps(models_with_features, device=device)
    
    print("\nDone!")
