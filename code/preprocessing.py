#!/usr/bin/env python3
"""
MAF Preprocessing Script

This script filters MAF files to identify bulk sequencing data by:
1. Checking for spatial/single-cell keywords in filenames
2. Checking for single-cell specific columns
3. Ensuring required bulk sequencing columns exist

Output: A markdown file with filtering results and statistics
"""

import pandas as pd
import numpy as np
import gzip
from pathlib import Path
from datetime import datetime

def find_maf_files(data_directory):
    """Find all MAF files in the data directory."""
    data_path = Path(data_directory)
    maf_files = []
    
    # Search for .maf and .maf.gz files
    for ext in ['*.maf', '*.maf.gz']:
        maf_files.extend(list(data_path.rglob(ext)))
    
    return maf_files

def read_maf_file(maf_file):
    """Read MAF file (handles both .maf and .maf.gz)."""
    try:
        if str(maf_file).endswith('.gz'):
            with gzip.open(maf_file, 'rt') as f:
                df = pd.read_csv(f, sep='\t', comment='#', low_memory=False)
        else:
            df = pd.read_csv(maf_file, sep='\t', comment='#', low_memory=False)
        return df
    except Exception as e:
        print(f"Error reading {maf_file.name}: {e}")
        return None

def is_bulk_sequencing(df, maf_file):
    """
    Check if the MAF file represents bulk sequencing data.
    
    Returns:
        (is_bulk, reason)
    """
    # Check for spatial/single-cell keywords in filename
    spatial_keywords = [
        'spatial', 'single-cell', 'sc-seq', 'visium', 
        'slide-seq', 'merfish', 'xenium', 'cosmx',
        '10x', 'cellranger', 'barcode'
    ]
    
    file_path_lower = str(maf_file).lower()
    for keyword in spatial_keywords:
        if keyword in file_path_lower:
            return False, f'Filename contains "{keyword}"'
    
    # Check for single-cell specific columns
    single_cell_columns = ['cell_barcode', 'barcode', 'x_position', 'y_position']
    for col in single_cell_columns:
        if col in df.columns:
            return False, f'Contains single-cell column "{col}"'
    
    # Check for required bulk sequencing columns
    required_columns = ['t_depth', 't_alt_count', 't_ref_count']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f'Missing required columns: {missing_columns}'
    
    return True, 'Bulk sequencing data'

def calculate_vaf_stats(df):
    """Calculate VAF statistics for the dataframe."""
    try:
        # Calculate VAF
        df['vaf'] = df['t_alt_count'] / df['t_depth']
        
        # Filter out invalid VAFs
        valid_vafs = df[(df['vaf'] > 0) & (df['vaf'] < 1) & (df['vaf'].notna())]['vaf']
        
        if len(valid_vafs) == 0:
            return None
        
        stats = {
            'n_mutations': len(valid_vafs),
            'vaf_mean': float(np.mean(valid_vafs)),
            'vaf_std': float(np.std(valid_vafs)),
            'vaf_min': float(np.min(valid_vafs)),
            'vaf_max': float(np.max(valid_vafs)),
            'depth_mean': float(np.mean(df['t_depth'])),
            'depth_median': float(np.median(df['t_depth']))
        }
        
        return stats
        
    except Exception as e:
        print(f"Error calculating VAF stats: {e}")
        return None

def extract_patient_uuid(maf_file):
    """Extract patient UUID from file path."""
    # Patient UUID is typically the directory name containing the MAF file
    return maf_file.parent.name

def process_files(data_directory):
    """Process all MAF files and categorize them."""
    print("🔍 Finding MAF files...")
    maf_files = find_maf_files(data_directory)
    print(f"Found {len(maf_files)} MAF files")
    
    bulk_files = []
    filtered_files = []
    
    print("\n📊 Processing files...")
    for i, maf_file in enumerate(maf_files, 1):
        if i % 100 == 0:
            print(f"Processed {i}/{len(maf_files)} files...")
        
        # Extract patient UUID
        patient_uuid = extract_patient_uuid(maf_file)
        
        # Read file
        df = read_maf_file(maf_file)
        if df is None:
            filtered_files.append({
                'patient_uuid': patient_uuid,
                'file_name': maf_file.name,
                'reason': 'Could not read file'
            })
            continue
        
        # Check if bulk sequencing
        is_bulk, reason = is_bulk_sequencing(df, maf_file)
        
        if is_bulk:
            # Calculate VAF statistics
            stats = calculate_vaf_stats(df)
            if stats:
                bulk_files.append({
                    'patient_uuid': patient_uuid,
                    'file_name': maf_file.name,
                    'file_path': str(maf_file),
                    'stats': stats
                })
            else:
                filtered_files.append({
                    'patient_uuid': patient_uuid,
                    'file_name': maf_file.name,
                    'reason': 'Could not calculate VAF statistics'
                })
        else:
            filtered_files.append({
                'patient_uuid': patient_uuid,
                'file_name': maf_file.name,
                'reason': reason
            })
    
    return bulk_files, filtered_files

def generate_markdown_report(bulk_files, filtered_files, output_file):
    """Generate markdown report with results."""
    
    # Calculate summary statistics
    total_files = len(bulk_files) + len(filtered_files)
    bulk_count = len(bulk_files)
    filtered_count = len(filtered_files)
    
    # Aggregate statistics for bulk files
    if bulk_files:
        all_stats = [f['stats'] for f in bulk_files]
        
        # Calculate averages
        avg_mutations = np.mean([s['n_mutations'] for s in all_stats])
        avg_vaf_mean = np.mean([s['vaf_mean'] for s in all_stats])
        avg_vaf_std = np.mean([s['vaf_std'] for s in all_stats])
        avg_depth = np.mean([s['depth_mean'] for s in all_stats])
        
        # Calculate ranges
        vaf_min_overall = min([s['vaf_min'] for s in all_stats])
        vaf_max_overall = max([s['vaf_max'] for s in all_stats])
        
        # Count rejection reasons
        rejection_reasons = {}
        for f in filtered_files:
            reason = f['reason']
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    else:
        avg_mutations = avg_vaf_mean = avg_vaf_std = avg_depth = 0
        vaf_min_overall = vaf_max_overall = 0
        rejection_reasons = {}
    
    # Generate markdown content
    md_content = f"""# MAF File Preprocessing Results

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total files processed**: {total_files}
- **Bulk sequencing patients**: {bulk_count} ({bulk_count/total_files*100:.1f}%)
- **Filtered out**: {filtered_count} ({filtered_count/total_files*100:.1f}%)

## Filtering Criteria

Files were filtered out if they contained:
- Spatial/single-cell keywords in filename: `spatial`, `single-cell`, `sc-seq`, `visium`, `slide-seq`, `merfish`, `xenium`, `cosmx`, `10x`, `cellranger`, `barcode`
- Single-cell specific columns: `cell_barcode`, `barcode`, `x_position`, `y_position`
- Missing required bulk columns: `t_depth`, `t_alt_count`, `t_ref_count`

## Bulk Files Statistics

### Average Statistics Across All Bulk Files
- **Average mutations per file**: {avg_mutations:.1f}
- **Average VAF mean**: {avg_vaf_mean:.3f}
- **Average VAF standard deviation**: {avg_vaf_std:.3f}
- **Average sequencing depth**: {avg_depth:.1f}x
- **Overall VAF range**: {vaf_min_overall:.3f} - {vaf_max_overall:.3f}

## Filtering Results

### Rejection Reasons
"""
    
    if rejection_reasons:
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
            md_content += f"- **{reason}**: {count} files\n"
    else:
        md_content += "No files were filtered out.\n"
    
    md_content += f"""
## Bulk Files List

Total: {bulk_count} patients

"""
    
    if bulk_files:
        # Sort by patient UUID
        sorted_files = sorted(bulk_files, key=lambda x: x['patient_uuid'])
        
        md_content += "### All Bulk Patients\n\n"
        for file_info in sorted_files:
            patient_uuid = file_info['patient_uuid']
            stats = file_info['stats']
            md_content += f"- **{patient_uuid}** ({stats['n_mutations']} mutations, VAF: {stats['vaf_min']:.3f}-{stats['vaf_max']:.3f})\n"
    else:
        md_content += "No bulk files found.\n"
    
    md_content += f"""
## Next Steps

These {bulk_count} bulk patients are ready for downstream analysis:
1. Neutrality testing
2. Coalescent tree reconstruction  
3. Spatial mapping

---
*Generated by preprocessing script*
"""
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(md_content)
    
    print(f"\n📄 Results saved to: {output_file}")

def main():
    """Main function."""
    # Configuration
    # Use relative paths from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_directory = os.path.join(project_root, "data", "snv maf")
    output_file = os.path.join(project_root, "results", "preprocessing_results.md")
    
    print("🚀 Starting MAF preprocessing...")
    print("=" * 60)
    
    # Process files
    bulk_files, filtered_files = process_files(data_directory)
    
    # Generate report
    generate_markdown_report(bulk_files, filtered_files, output_file)
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ Preprocessing complete!")
    print(f"📊 {len(bulk_files)} bulk files identified")
    print(f"❌ {len(filtered_files)} files filtered out")
    print(f"📄 Results saved to: {output_file}")

if __name__ == "__main__":
    main()
