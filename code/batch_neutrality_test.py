#!/usr/bin/env python3
"""
Batch neutrality testing for multiple MAF files.
Processes all bulk MAF files identified during preprocessing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import glob
import json
from pathlib import Path

def test_neutral_evolution(maf_file, min_vaf=0.12, max_vaf=0.40, min_depth=10, min_alt_reads=3):
    """
    Test if tumor evolution follows neutral model using 1/f power law.
    
    Parameters:
    - maf_file: Path to MAF file
    - min_vaf, max_vaf: VAF range for subclonal mutations
    - min_depth: Minimum sequencing depth
    - min_alt_reads: Minimum alternate allele reads
    
    Returns:
    - Dictionary with results
    """
    
    try:
        # Load and filter mutations
        df = pd.read_csv(maf_file, sep='\t', comment='#')
        
        # Calculate VAF
        df['vaf'] = df['t_alt_count'] / df['t_depth']
        
        # Filter for subclonal mutations
        subclonal_mask = (
            (df['vaf'] >= min_vaf) & 
            (df['vaf'] <= max_vaf) &
            (df['t_depth'] >= min_depth) &
            (df['t_alt_count'] >= min_alt_reads)
        )
        
        subclonal_mutations = df[subclonal_mask].copy()
        
        if len(subclonal_mutations) < 10:
            return {
                'status': 'insufficient_data',
                'n_subclonal': len(subclonal_mutations),
                'r_squared': None,
                'is_neutral': False,
                'error': f'Only {len(subclonal_mutations)} subclonal mutations found'
            }
        
        # Sort by VAF (descending)
        subclonal_mutations = subclonal_mutations.sort_values('vaf', ascending=False)
        
        # Calculate cumulative mutations M(f) using VAF directly
        subclonal_mutations['cumulative_mutations'] = range(1, len(subclonal_mutations) + 1)
        
        # Linear regression: M(f) = slope * (1/f) + intercept
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            subclonal_mutations['vaf'], 
            subclonal_mutations['cumulative_mutations']
        )
        
        r_squared = r_value ** 2
        is_neutral = bool(r_squared >= 0.98)
        
        return {
            'status': 'success',
            'n_subclonal': len(subclonal_mutations),
            'r_squared': float(r_squared),
            'slope': float(slope),
            'p_value': float(p_value),
            'is_neutral': is_neutral,
            'vaf_range': (float(min_vaf), float(max_vaf)),
            'min_depth': int(min_depth),
            'min_alt_reads': int(min_alt_reads)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'r_squared': None,
            'is_neutral': False
        }

def create_individual_plot(maf_file, results, output_dir):
    """
    Create individual neutrality test plot for a single MAF file.
    """
    try:
        # Load data for plotting
        df = pd.read_csv(maf_file, sep='\t', comment='#')
        df['vaf'] = df['t_alt_count'] / df['t_depth']
        
        # Apply same filters as in test
        subclonal_mask = (
            (df['vaf'] >= results['vaf_range'][0]) & 
            (df['vaf'] <= results['vaf_range'][1]) &
            (df['t_depth'] >= results['min_depth']) &
            (df['t_alt_count'] >= results['min_alt_reads'])
        )
        
        subclonal_mutations = df[subclonal_mask].copy()
        subclonal_mutations = subclonal_mutations.sort_values('vaf', ascending=False)
        subclonal_mutations['cumulative_mutations'] = range(1, len(subclonal_mutations) + 1)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(subclonal_mutations['vaf'], subclonal_mutations['cumulative_mutations'], 
                    alpha=0.6, s=50, label='Observed data')
        
        # Plot linear fit
        if results['r_squared'] is not None:
            slope, intercept, _, _, _ = stats.linregress(
                subclonal_mutations['vaf'], 
                subclonal_mutations['cumulative_mutations']
            )
            x_fit = np.linspace(subclonal_mutations['vaf'].min(), 
                               subclonal_mutations['vaf'].max(), 100)
            y_fit = slope * x_fit + intercept
            plt.plot(x_fit, y_fit, 'r-', linewidth=2, 
                     label=f'Linear fit (R² = {results["r_squared"]:.4f})')
        
        plt.xlabel('VAF (Variant Allele Frequency)')
        plt.ylabel('Cumulative Number of Mutations M(f)')
        
        # Extract patient UUID from file path
        patient_uuid = Path(maf_file).parent.name
        plt.title(f'Neutrality Test - Patient {patient_uuid}')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add neutrality assessment
        status = "NEUTRAL" if results['is_neutral'] else "NON-NEUTRAL"
        color = "green" if results['is_neutral'] else "red"
        
        plt.text(0.05, 0.95, f'Status: {status}\nR² = {results["r_squared"]:.4f}', 
                 transform=plt.gca().transAxes, fontsize=12, 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3),
                 verticalalignment='top')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"neutrality_test_{patient_uuid}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"Error creating plot for {maf_file}: {e}")
        return None

def process_batch_neutrality_testing(data_dir, results_dir):
    """
    Process all MAF files for neutrality testing.
    """
    print("🔬 Starting batch neutrality testing...")
    print("=" * 50)
    
    # Find all MAF files
    maf_pattern = os.path.join(data_dir, "**", "*.maf")
    maf_files = glob.glob(maf_pattern, recursive=True)
    
    # Also find .maf.gz files
    maf_gz_pattern = os.path.join(data_dir, "**", "*.maf.gz")
    maf_gz_files = glob.glob(maf_gz_pattern, recursive=True)
    
    all_maf_files = maf_files + maf_gz_files
    
    print(f"Found {len(all_maf_files)} MAF files")
    
    # Create neutrality testing results directory
    neutrality_dir = os.path.join(results_dir, "neutrality_tests")
    os.makedirs(neutrality_dir, exist_ok=True)
    
    # Process each file
    results_summary = []
    neutral_count = 0
    non_neutral_count = 0
    error_count = 0
    
    for i, maf_file in enumerate(all_maf_files, 1):
        print(f"Processing {i}/{len(all_maf_files)}: {os.path.basename(maf_file)}")
        
        # Extract patient UUID from file path
        patient_uuid = Path(maf_file).parent.name
        
        # Run neutrality test
        results = test_neutral_evolution(maf_file)
        
        # Add file information
        results['maf_file'] = maf_file
        results['patient_uuid'] = patient_uuid
        results['filename'] = os.path.basename(maf_file)
        
        # Create individual plot if successful
        if results['status'] == 'success':
            plot_path = create_individual_plot(maf_file, results, neutrality_dir)
            results['plot_path'] = plot_path
        
        # Count results
        if results['status'] == 'success':
            if results['is_neutral']:
                neutral_count += 1
            else:
                non_neutral_count += 1
        else:
            error_count += 1
        
        results_summary.append(results)
    
    # Save detailed results
    results_file = os.path.join(neutrality_dir, "neutrality_test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Create summary report
    create_summary_report(results_summary, neutrality_dir)
    
    print("\n" + "=" * 50)
    print("✅ Batch neutrality testing complete!")
    print(f"📊 Results summary:")
    print(f"   • Neutral tumors: {neutral_count}")
    print(f"   • Non-neutral tumors: {non_neutral_count}")
    print(f"   • Errors/insufficient data: {error_count}")
    print(f"   • Total processed: {len(all_maf_files)}")
    print(f"📄 Detailed results saved to: {results_file}")
    print(f"📊 Summary report saved to: {os.path.join(neutrality_dir, 'neutrality_summary.md')}")
    
    return results_summary

def create_summary_report(results_summary, output_dir):
    """
    Create a markdown summary report of neutrality testing results.
    """
    report_path = os.path.join(output_dir, "neutrality_summary.md")
    
    # Count results
    neutral_count = sum(1 for r in results_summary if r.get('is_neutral', False))
    non_neutral_count = sum(1 for r in results_summary if r.get('status') == 'success' and not r.get('is_neutral', True))
    error_count = sum(1 for r in results_summary if r.get('status') != 'success')
    total_count = len(results_summary)
    
    with open(report_path, 'w') as f:
        f.write("# Neutrality Testing Summary Report\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- **Total MAF files processed:** {total_count}\n")
        f.write(f"- **Neutral tumors:** {neutral_count} ({neutral_count/total_count*100:.1f}%)\n")
        f.write(f"- **Non-neutral tumors:** {non_neutral_count} ({non_neutral_count/total_count*100:.1f}%)\n")
        f.write(f"- **Errors/insufficient data:** {error_count} ({error_count/total_count*100:.1f}%)\n\n")
        
        f.write("## Testing Parameters\n\n")
        f.write("- **VAF range:** 0.12 - 0.40\n")
        f.write("- **Minimum depth:** 10\n")
        f.write("- **Minimum alt reads:** 3\n")
        f.write("- **Neutrality threshold:** R² ≥ 0.98\n\n")
        
        f.write("## Results by Patient\n\n")
        f.write("| Patient UUID | Status | R² | Subclonal Mutations | Notes |\n")
        f.write("|--------------|--------|----|---------------------|-------|\n")
        
        for result in results_summary:
            patient_uuid = result['patient_uuid']
            status = "NEUTRAL" if result.get('is_neutral', False) else "NON-NEUTRAL" if result.get('status') == 'success' else "ERROR"
            r_squared = f"{result.get('r_squared', 0):.4f}" if result.get('r_squared') else "N/A"
            n_subclonal = result.get('n_subclonal', 0)
            notes = result.get('error', '') if result.get('status') != 'success' else ''
            
            f.write(f"| {patient_uuid} | {status} | {r_squared} | {n_subclonal} | {notes} |\n")
        
        f.write("\n## Interpretation\n\n")
        f.write("**Neutral tumors (R² ≥ 0.98):** These tumors show evidence of neutral evolution, ")
        f.write("suggesting that VAF can be used as a reliable proxy for mutation timing in coalescent analysis.\n\n")
        f.write("**Non-neutral tumors (R² < 0.98):** These tumors show evidence of non-neutral evolution, ")
        f.write("indicating that clonal selection or other evolutionary forces may be affecting the mutation distribution.\n\n")
        f.write("**Errors/Insufficient data:** Files that could not be processed due to technical issues ")
        f.write("or insufficient subclonal mutations (< 10) for reliable analysis.\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python batch_neutrality_test.py <data_directory> <results_directory>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    results_dir = sys.argv[2]
    
    process_batch_neutrality_testing(data_dir, results_dir)
