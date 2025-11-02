#!/usr/bin/env python3
"""
Batch spatial mapping script for coalescent trees.
Applies Gaussian diffusion to assign spatial positions to tree nodes,
partitions lineages, and calculates spatial metrics.
"""

import os
import sys
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
import scipy.stats as stats
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

# Import coalescent tree reconstruction functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_coalescent_reconstruction import (
    load_maf_data,
    estimate_effective_population_size,
    generate_coalescent_process,
    build_coalescent_tree_structure,
    map_mutations_to_lineages
)


def set_coalescent_node_positions(tree_structure, root_node, seed=None, noise_scale=1.0):
    """
    Set node positions using Gaussian diffusion, adapted for our coalescent tree structure.
    Works from root to leaves with cumulative noise.
    
    Args:
        tree_structure: Dictionary representing the coalescent tree
        root_node: Name of the root node
        seed: Random seed for reproducibility
        noise_scale: Factor to control the overall amount of noise
    
    Returns:
        Dictionary mapping node names to (x, y) positions
    """
    if seed is not None:
        np.random.seed(seed)
    
    positions = {}
    
    # Start with root at origin
    positions[root_node] = np.array([0.0, 0.0])
    
    # Function to process each node recursively
    def place_children(node_name):
        if node_name not in tree_structure:
            return
        
        node_data = tree_structure[node_name]
        children = node_data.get('children', [])
        
        if not children:  # Leaf node
            return
        
        # Calculate fan-out angle based on number of children
        if len(children) > 1:
            # Use wider angles for nodes with more children
            angles = np.linspace(-np.pi/3, np.pi/3, len(children))
        else:
            # Single child - add small random angle
            angles = [np.random.uniform(-np.pi/8, np.pi/8)]
        
        for i, child_name in enumerate(children):
            if child_name not in tree_structure:
                continue
            
            child_data = tree_structure[child_name]
            edge_length = child_data.get('edge_length', 0.0)
            
            # Handle zero edge lengths
            if edge_length <= 0:
                edge_length = 0.001  # Small default value
            
            # Base displacement - move away from parent
            distance = np.sqrt(edge_length) * noise_scale
            
            # Calculate direction
            parent_name = tree_structure[node_name].get('parent')
            if parent_name is None or parent_name not in positions:
                # Root's children scatter in all directions
                angle = angles[i] if len(children) > 1 else np.random.uniform(-np.pi/3, np.pi/3)
            else:
                # Get parent direction and add child's fan-out angle
                parent_pos = positions.get(parent_name, positions[node_name])
                parent_vector = np.array(positions[node_name]) - np.array(parent_pos)
                if np.linalg.norm(parent_vector) > 0:
                    parent_angle = np.arctan2(parent_vector[1], parent_vector[0])
                else:
                    parent_angle = 0
                
                # Combine parent direction with fan-out
                angle = parent_angle + angles[i]
            
            # Apply base vector
            dx = distance * np.cos(angle)
            dy = distance * np.sin(angle)
            
            # Add random noise proportional to branch length
            noise_std = np.sqrt(edge_length) * 0.3
            if noise_std <= 0:
                noise_std = 0.001
            noise = np.random.normal(0, noise_std, 2)
            
            # Position = parent position + direction vector + noise
            parent_pos = positions[node_name]
            positions[child_name] = parent_pos + np.array([dx, dy]) + noise
            
            # Recursively position this node's children
            place_children(child_name)
    
    # Start the recursive placement from the root
    place_children(root_node)
    
    return positions


def binary_lineage_partitioning(tree_structure, internal_nodes, leaf_nodes):
    """
    Partition the tree into 2 lineages based on the first major coalescent event.
    This finds the two largest subtrees from the root.
    
    Args:
        tree_structure: Dictionary representing the coalescent tree
        internal_nodes: List of internal node names
        leaf_nodes: List of leaf node names
    
    Returns:
        node_colors: Dictionary mapping node names to lineage IDs (0=root, 1=lineage1, 2=lineage2)
        founders: List of founder node names (the two main lineage founders)
    """
    if not internal_nodes:
        # No internal nodes - all nodes are in lineage 1
        return {node: 1 for node in leaf_nodes}, []
    
    root_node = internal_nodes[-1]  # Root is the last coalescent event
    root_children = tree_structure[root_node].get('children', [])
    
    if len(root_children) < 2:
        # Degenerate case - assign everything to one lineage
        all_nodes = internal_nodes + leaf_nodes
        return {node: 1 for node in all_nodes}, root_children
    
    # In a binary coalescent tree, the root should have exactly 2 children
    # Both children are the founders of the two main lineages
    if len(root_children) != 2:
        # Unexpected case: more than 2 children (shouldn't happen in binary tree)
        # Take first two as founders
        founders = root_children[:2]
    else:
        founders = root_children
    
    # Get descendants for each founder
    def get_descendants(node_name):
        """Get all descendant nodes (recursive), including the node itself."""
        descendants = set([node_name])
        for child in tree_structure.get(node_name, {}).get('children', []):
            descendants.update(get_descendants(child))
        return descendants
    
    # Get descendants for each founder BEFORE assignment to verify partitioning
    founder1_desc = get_descendants(founders[0])
    founder2_desc = get_descendants(founders[1])
    
    # Verify no overlap (shouldn't happen in a binary tree, but check anyway)
    overlap = founder1_desc & founder2_desc
    if overlap and overlap != {root_node}:
        # Overlap is allowed to include root_node only
        non_root_overlap = overlap - {root_node}
        if non_root_overlap:
            print(f"  Warning: {len(non_root_overlap)} nodes found in both lineages: {list(non_root_overlap)[:5]}")
    
    # Verify all nodes are covered (except root)
    all_nodes = set(internal_nodes + leaf_nodes) - {root_node}
    covered = (founder1_desc | founder2_desc) - {root_node}
    missing = all_nodes - covered
    if missing:
        print(f"  Warning: {len(missing)} nodes not in either lineage's descendants: {list(missing)[:5]}")
    
    # Assign lineage colors
    node_colors = {}
    
    # Root gets color 0
    node_colors[root_node] = 0
    
    # Assign lineage 1 (first founder and all its descendants)
    for node in founder1_desc:
        if node != root_node:  # Don't overwrite root color
            node_colors[node] = 1
    
    # Assign lineage 2 (second founder and all its descendants)
    # This assignment happens after lineage 1, so if there's any overlap, lineage 2 will overwrite
    for node in founder2_desc:
        if node != root_node:  # Don't overwrite root color
            node_colors[node] = 2
    
    # Final verification: count lineage sizes
    lineage1_count = sum(1 for v in node_colors.values() if v == 1)
    lineage2_count = sum(1 for v in node_colors.values() if v == 2)
    
    # Verify the partitioning is correct
    total_assigned = lineage1_count + lineage2_count
    total_nodes = len(internal_nodes) + len(leaf_nodes) - 1  # Excluding root
    
    # In a proper binary tree, all nodes (except root) should be assigned
    if total_assigned != total_nodes:
        print(f"  Warning: Only {total_assigned}/{total_nodes} nodes assigned to lineages")
    
    # Log lineage sizes for debugging (only if extreme imbalance)
    if total_assigned > 0:
        lineage1_ratio = lineage1_count / total_assigned
        lineage2_ratio = lineage2_count / total_assigned
        
        # Note: In random coalescent trees, one subtree can legitimately have 
        # many more nodes than the other. This is expected behavior.
        if abs(lineage1_ratio - lineage2_ratio) > 0.7:  # >85% imbalance
            # This is a significant imbalance, but it's expected in stochastic coalescent processes
            # The tree structure is correct - it's just that one lineage happened to 
            # accumulate more mutations/lineages through random coalescence
            pass
    
    # Handle any remaining unassigned nodes (shouldn't happen)
    all_nodes_check = set(internal_nodes + leaf_nodes) - {root_node}
    unassigned = all_nodes_check - set(node_colors.keys())
    if unassigned:
        print(f"  Warning: {len(unassigned)} nodes not assigned to any lineage")
        # Assign to lineage 1 as fallback
        for node in unassigned:
            node_colors[node] = 1
    
    return node_colors, founders


def calculate_morans_i_coalescent(positions, node_colors, exclude_value=None, knn=10):
    """
    Calculate Moran's I spatial autocorrelation statistic for coalescent trees.
    
    Args:
        positions: Dictionary mapping node names to (x, y) positions
        node_colors: Dictionary mapping node names to lineage IDs
        exclude_value: Value to exclude from calculation (e.g., root partition)
        knn: Number of nearest neighbors to consider
    
    Returns:
        I: Moran's I statistic
        p_value: p-value for the statistic
    """
    # Convert to arrays
    node_names = list(positions.keys())
    pos_array = np.array([positions[node] for node in node_names])
    color_array = np.array([node_colors.get(node, 0) for node in node_names])
    
    # Filter if excluding a value
    if exclude_value is not None:
        mask = color_array != exclude_value
        if np.sum(mask) <= 1:
            return 0.0, 1.0
        pos_array = pos_array[mask]
        color_array = color_array[mask]
    
    if len(pos_array) <= 1:
        return 0.0, 1.0
    
    # Create weights matrix based on k-nearest neighbors (original method for all cases)
    distances = squareform(pdist(pos_array))
    weights = np.zeros_like(distances)
    
    n_nodes = len(pos_array)
    for i in range(n_nodes):
        effective_knn = min(knn, n_nodes - 1)
        nearest = np.argsort(distances[i])[1:effective_knn+1]
        weights[i, nearest] = 1
    
    # Standardize weights matrix by row
    row_sums = weights.sum(axis=1)
    non_zero_rows = row_sums > 0
    if np.any(non_zero_rows):
        weights[non_zero_rows] = weights[non_zero_rows] / row_sums[non_zero_rows, np.newaxis]
    
    # Calculate Moran's I
    n = len(color_array)
    if n <= 1:
        return 0.0, 1.0
    
    z = color_array - np.mean(color_array)
    z_2 = np.sum(z**2)
    
    if z_2 == 0:
        return 0.0, 1.0
    
    # Calculate the numerator (spatial autocovariance) using nested loops
    numerator = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                numerator += weights[i, j] * z[i] * z[j]
    
    # Calculate Moran's I
    total_weights = np.sum(weights)
    if total_weights > 0 and z_2 > 0:
        I = (n / total_weights) * (numerator / z_2)
    else:
        return 0.0, 1.0
    
    # Calculate p-value
    E_I = -1.0 / (n - 1)
    var_I = 1.0 / (n - 1)
    z_score = (I - E_I) / np.sqrt(var_I) if var_I > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    return float(I), float(p_value)


def calculate_spatial_metrics_coalescent(positions, node_colors, founders):
    """
    Calculate within-lineage spread (alpha) and founder separation (beta) metrics.
    
    Args:
        positions: Dictionary mapping node names to (x, y) positions
        node_colors: Dictionary mapping node names to lineage IDs
        founders: List of founder node names
    
    Returns:
        alpha: Average spread of nodes within each lineage
        beta: Average pairwise distance between founders
    """
    # Calculate within-lineage spread (alpha)
    lineage_spreads = []
    
    # Get unique lineages (excluding root partition 0)
    unique_lineages = set(node_colors.values()) - {0}
    
    for lineage_id in unique_lineages:
        # Get all nodes in this lineage
        lineage_nodes = [node for node, color in node_colors.items() 
                        if color == lineage_id and node in positions]
        
        if len(lineage_nodes) < 2:
            continue
        
        # Get positions
        lineage_positions = np.array([positions[node] for node in lineage_nodes])
        
        # Calculate centroid
        centroid = np.mean(lineage_positions, axis=0)
        
        # Calculate average distance from centroid
        distances = np.linalg.norm(lineage_positions - centroid, axis=1)
        avg_spread = np.mean(distances)
        lineage_spreads.append(avg_spread)
    
    # Average spread across all lineages
    alpha = np.mean(lineage_spreads) if lineage_spreads else 0.0
    
    # Calculate founder separation (beta)
    if len(founders) < 2:
        beta = 0.0
    else:
        # Get founder positions
        founder_positions = np.array([positions[f] for f in founders if f in positions])
        
        if len(founder_positions) >= 2:
            # Calculate pairwise distances
            pairwise_distances = pdist(founder_positions)
            beta = np.mean(pairwise_distances)
        else:
            beta = 0.0
    
    return float(alpha), float(beta)


def visualize_coalescent_spatial_tree(tree_structure, positions, node_colors, founders,
                                     mutation_mapping, patient_uuid, output_dir,
                                     morans_i, p_value, alpha, beta):
    """
    Create visualization of the spatial mapping with metrics displayed separately.
    """
    try:
        # Filter out root (partition 0) for visualization
        plot_nodes = {node: pos for node, pos in positions.items() 
                     if node_colors.get(node, 0) != 0}
        plot_colors = {node: color for node, color in node_colors.items() 
                      if color != 0}
        
        if not plot_nodes:
            return None
        
        # Convert to arrays
        node_names = list(plot_nodes.keys())
        pos_array = np.array([plot_nodes[node] for node in node_names])
        color_array = np.array([plot_colors[node] for node in node_names])
        
        # Create figure with side panel for metrics
        fig = plt.figure(figsize=(16, 10))
        
        # Main plot (left side)
        ax_main = plt.subplot(1, 2, 1)
        
        # Get lineage IDs
        lineage1_nodes = [node for node in node_names if color_array[node_names.index(node)] == 1]
        lineage2_nodes = [node for node in node_names if color_array[node_names.index(node)] == 2]
        
        # Plot in the exact same order and format as coalescent_spatial_mapping.py
        colors = ['red', 'blue']  # Lineage 1 and Lineage 2
        
        # Plot internal nodes (non-leaf nodes) first
        for node_name in plot_nodes:
            if node_name not in tree_structure:
                continue
            
            node_data = tree_structure[node_name]
            
            # Skip leaf nodes for now (will plot separately)
            if node_data.get('type') == 'lineage':
                continue
            
            x, y = plot_nodes[node_name]
            color_id = node_colors.get(node_name, 0)
            
            if color_id == 0:  # Root node
                # Root node - large black circle
                ax_main.scatter(x, y, c='black', s=150, alpha=0.8, marker='o', 
                          edgecolors='white', linewidth=2)
            elif color_id in [1, 2]:  # Internal lineage nodes
                # Internal lineage nodes - colored circles with white edges
                ax_main.scatter(x, y, c=colors[color_id-1], s=50, alpha=0.8, marker='o',
                          edgecolors='white', linewidth=1)
        
        # Plot leaf nodes with special colors (mutations)
        for node_name in plot_nodes:
            if node_name not in tree_structure:
                continue
            
            node_data = tree_structure[node_name]
            
            # Only plot leaf nodes (mutations)
            if node_data.get('type') != 'lineage':
                continue
            
            x, y = plot_nodes[node_name]
            color_id = node_colors.get(node_name, 0)
            
            if color_id == 1:  # Lineage 1 mutations
                ax_main.scatter(x, y, c='pink', s=80, alpha=0.8, marker='o',
                          edgecolors='red', linewidth=1.5)
            elif color_id == 2:  # Lineage 2 mutations
                ax_main.scatter(x, y, c='lightblue', s=80, alpha=0.8, marker='o',
                          edgecolors='blue', linewidth=1.5)
        
        # Highlight founders with colored squares
        founder_colors = ['red', 'blue']
        for i, founder in enumerate(founders):
            if founder in plot_nodes:
                x, y = plot_nodes[founder]
                ax_main.scatter(x, y, c=founder_colors[i], s=200, marker='s', alpha=0.9,
                          edgecolors='black', linewidth=2, zorder=10)
        
        # Add legend matching original format
        legend_elements = []
        legend_elements.append(Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='none', label='Lineage 1'))
        legend_elements.append(Rectangle((0, 0), 1, 1, facecolor='blue', edgecolor='none', label='Lineage 2'))
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                     markerfacecolor='red', markersize=12, 
                                     linestyle='', label='Founder 1'))
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                     markerfacecolor='blue', markersize=12, 
                                     linestyle='', label='Founder 2'))
        
        ax_main.legend(handles=legend_elements, loc='upper right', fontsize=9)
        ax_main.set_xlabel('X Position', fontsize=12)
        ax_main.set_ylabel('Y Position', fontsize=12)
        ax_main.set_title(f'Spatial Mapping - Patient {patient_uuid}', fontsize=14, fontweight='bold')
        ax_main.set_aspect('equal')
        ax_main.grid(True, alpha=0.3)
        
        # Metrics panel (right side)
        ax_metrics = plt.subplot(1, 2, 2)
        ax_metrics.axis('off')
        
        # Calculate ratio
        ratio = beta / alpha if alpha > 0 else 0.0
        
        # Create text box with metrics
        metrics_text = f"""
Spatial Metrics

Moran's I: {morans_i:.4f}
p-value: {p_value:.4f}

Within-lineage Spread (α): {alpha:.4f}
Founder Separation (β): {beta:.4f}

Ratio (β/α): {ratio:.4f}
        """
        
        ax_metrics.text(0.1, 0.5, metrics_text.strip(), 
                       fontsize=12, verticalalignment='center',
                       family='monospace', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, f'spatial_mapping_{patient_uuid}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    except Exception as e:
        print(f"  Error creating spatial plot: {str(e)}")
        plt.close()
        return None


def process_single_tree_for_spatial_mapping(maf_file, output_dir, coalescent_result=None, random_seed=None):
    """
    Process a single MAF file to create spatial mapping of the coalescent tree.
    
    Args:
        maf_file: Path to MAF file
        output_dir: Output directory for results
        coalescent_result: Optional coalescent result dict from JSON (if None, will regenerate)
        random_seed: Optional random seed (if None and coalescent_result provided, uses saved seed)
    
    Returns:
        Dictionary with results or error information
    """
    try:
        # Extract patient UUID from file path
        patient_uuid = Path(maf_file).parent.name
        
        # Step 1: Load MAF data
        mutations, error = load_maf_data(maf_file)
        if error:
            return {
                'status': 'error',
                'patient_uuid': patient_uuid,
                'maf_file': maf_file,
                'error': error
            }
        
        # Step 2: Use coalescent tree from saved results if available
        if coalescent_result is not None and 'coalescent_events' in coalescent_result:
            # Use saved coalescent events and seed from JSON
            saved_events = coalescent_result['coalescent_events']
            saved_seed = coalescent_result.get('random_seed', None)
            saved_ne = coalescent_result.get('effective_population_size', None)
            saved_height = coalescent_result.get('total_tree_height', None)
            
            # Use saved seed or derive from UUID
            if saved_seed is not None:
                random_seed = int(saved_seed)
            elif random_seed is None:
                hash_obj = hashlib.md5(patient_uuid.encode('utf-8'))
                random_seed = int(hash_obj.hexdigest(), 16) % (2**31)
            
            # Reconstruct coalescent events from saved data
            coalescent_events = saved_events  # Already in correct format
            total_height = float(saved_height) if saved_height is not None else sum(e['waiting_time'] for e in saved_events)
            
            # Use saved Ne to ensure consistency (should match calculated Ne)
            if saved_ne is not None:
                Ne = float(saved_ne)
            else:
                # Fallback: calculate from VAFs
                vafs = [m['vaf'] for m in mutations]
                Ne = estimate_effective_population_size(vafs)
            
            # Verify Ne matches calculated value (for validation)
            vafs = [m['vaf'] for m in mutations]
            calculated_ne = estimate_effective_population_size(vafs)
            if abs(Ne - calculated_ne) > 0.01:
                print(f"  Warning: Ne mismatch (saved={Ne:.4f}, calculated={calculated_ne:.4f}), using saved value")
            
            # Step 3: Build coalescent tree structure using saved events
            # Use same seed to ensure deterministic lineage selection
            np.random.seed(random_seed)
            tree_structure, internal_nodes, leaf_nodes = build_coalescent_tree_structure(
                coalescent_events, len(mutations))
        else:
            # Fallback: regenerate coalescent process (for backward compatibility)
            # Use deterministic seed based on patient UUID for reproducibility
            if random_seed is None:
                hash_obj = hashlib.md5(patient_uuid.encode('utf-8'))
                random_seed = int(hash_obj.hexdigest(), 16) % (2**31)
            
            # Step 2: Estimate effective population size
            vafs = [m['vaf'] for m in mutations]
            Ne = estimate_effective_population_size(vafs)
            
            # Step 3: Generate coalescent process (with seed for reproducibility)
            n_lineages = len(mutations)
            np.random.seed(random_seed)
            coalescent_events, total_height = generate_coalescent_process(n_lineages, Ne, random_seed)
            
            # Step 4: Build coalescent tree structure
            np.random.seed(random_seed)
            tree_structure, internal_nodes, leaf_nodes = build_coalescent_tree_structure(
                coalescent_events, n_lineages)
        
        # Step 5: Map mutations to lineages
        mutation_mapping = map_mutations_to_lineages(mutations, leaf_nodes, tree_structure)
        
        # Step 6: Apply spatial positioning
        root_node = internal_nodes[-1] if internal_nodes else leaf_nodes[0]
        positions = set_coalescent_node_positions(tree_structure, root_node, 
                                                  seed=random_seed, noise_scale=1.0)
        
        # Step 7: Partition lineages
        node_colors, founders = binary_lineage_partitioning(
            tree_structure, internal_nodes, leaf_nodes)
        
        # Step 8: Calculate spatial metrics
        morans_i, p_value = calculate_morans_i_coalescent(positions, node_colors, exclude_value=0)
        alpha, beta = calculate_spatial_metrics_coalescent(positions, node_colors, founders)
        
        # Step 9: Create visualization
        plot_path = visualize_coalescent_spatial_tree(
            tree_structure, positions, node_colors, founders, mutation_mapping,
            patient_uuid, output_dir, morans_i, p_value, alpha, beta)
        
        # Compile results
        results = {
            'status': 'success',
            'patient_uuid': patient_uuid,
            'maf_file': maf_file,
            'n_mutations': len(mutations),
            'effective_population_size': float(Ne),
            'total_tree_height': float(total_height),
            'spatial_metrics': {
                'morans_i': float(morans_i),
                'morans_i_p_value': float(p_value),
                'within_lineage_spread_alpha': float(alpha),
                'founder_separation_beta': float(beta),
                'beta_alpha_ratio': float(beta/alpha) if alpha > 0 else 0.0
            },
            'n_lineages': len(set(node_colors.values()) - {0}),
            'plot_path': plot_path
        }
        
        return results
        
    except Exception as e:
        return {
            'status': 'error',
            'patient_uuid': Path(maf_file).parent.name,
            'maf_file': maf_file,
            'error': str(e)
        }


def create_summary_report(results_summary, output_dir, alpha_values=None, beta_values=None):
    """Create a markdown summary report with distribution plots."""
    output_file = os.path.join(output_dir, "spatial_summary.md")
    
    # Create distribution plots if data is provided
    if alpha_values is not None and beta_values is not None and len(alpha_values) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Alpha distribution
        axes[0].hist(alpha_values, bins=30, color='blue', alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(alpha_values), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(alpha_values):.4f}')
        axes[0].axvline(np.median(alpha_values), color='green', linestyle='--', linewidth=2, 
                      label=f'Median: {np.median(alpha_values):.4f}')
        axes[0].set_xlabel('Within-lineage Spread (α)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Within-lineage Spread (α)', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Beta distribution
        axes[1].hist(beta_values, bins=30, color='red', alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(beta_values), color='blue', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(beta_values):.4f}')
        axes[1].axvline(np.median(beta_values), color='green', linestyle='--', linewidth=2, 
                      label=f'Median: {np.median(beta_values):.4f}')
        axes[1].set_xlabel('Founder Separation (β)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Distribution of Founder Separation (β)', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        dist_plot_path = os.path.join(output_dir, "spatial_metrics_distributions.png")
        plt.savefig(dist_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    with open(output_file, 'w') as f:
        f.write("# Spatial Mapping Summary Report\n\n")
        f.write(f"**Generated:** {results_summary.get('timestamp', 'N/A')}\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- **Total trees processed:** {results_summary.get('total_processed', 0)}\n")
        f.write(f"- **Successful mappings:** {results_summary.get('success_count', 0)} "
               f"({100*results_summary.get('success_count', 0)/max(results_summary.get('total_processed', 1), 1):.1f}%)\n")
        f.write(f"- **Errors:** {results_summary.get('error_count', 0)}\n\n")
        
        if results_summary.get('success_count', 0) > 0:
            f.write("## Spatial Metrics Statistics\n\n")
            metrics = results_summary.get('metrics_summary', {})
            f.write(f"- **Average Moran's I:** {metrics.get('mean_morans_i', 0):.4f} "
                   f"± {metrics.get('std_morans_i', 0):.4f}\n")
            f.write(f"- **Average within-lineage spread (α):** {metrics.get('mean_alpha', 0):.4f} "
                   f"± {metrics.get('std_alpha', 0):.4f}\n")
            f.write(f"- **Average founder separation (β):** {metrics.get('mean_beta', 0):.4f} "
                   f"± {metrics.get('std_beta', 0):.4f}\n")
            f.write(f"- **Average β/α ratio:** {metrics.get('mean_ratio', 0):.4f} "
                   f"± {metrics.get('std_ratio', 0):.4f}\n\n")
            
            if alpha_values is not None and beta_values is not None and len(alpha_values) > 0:
                f.write("## Distribution Statistics\n\n")
                f.write(f"### Within-lineage Spread (α)\n\n")
                f.write(f"- **Mean:** {np.mean(alpha_values):.4f}\n")
                f.write(f"- **Median:** {np.median(alpha_values):.4f}\n")
                f.write(f"- **Standard Deviation:** {np.std(alpha_values):.4f}\n")
                f.write(f"- **Min:** {np.min(alpha_values):.4f}\n")
                f.write(f"- **Max:** {np.max(alpha_values):.4f}\n")
                f.write(f"- **25th Percentile:** {np.percentile(alpha_values, 25):.4f}\n")
                f.write(f"- **75th Percentile:** {np.percentile(alpha_values, 75):.4f}\n\n")
                
                f.write(f"### Founder Separation (β)\n\n")
                f.write(f"- **Mean:** {np.mean(beta_values):.4f}\n")
                f.write(f"- **Median:** {np.median(beta_values):.4f}\n")
                f.write(f"- **Standard Deviation:** {np.std(beta_values):.4f}\n")
                f.write(f"- **Min:** {np.min(beta_values):.4f}\n")
                f.write(f"- **Max:** {np.max(beta_values):.4f}\n")
                f.write(f"- **25th Percentile:** {np.percentile(beta_values, 25):.4f}\n")
                f.write(f"- **75th Percentile:** {np.percentile(beta_values, 75):.4f}\n\n")
                
                f.write("## Distribution Plots\n\n")
                f.write("![Spatial Metrics Distributions](spatial_metrics_distributions.png)\n\n")
                f.write("*Figure: Distribution of within-lineage spread (α) and founder separation (β) across all trees.*\n\n")


def main(data_directory, coalescent_results_file, results_directory):
    """
    Main function to process all coalescent trees for spatial mapping.
    """
    from datetime import datetime
    
    # Create output directory
    spatial_dir = os.path.join(results_directory, 'spatial_mapping')
    os.makedirs(spatial_dir, exist_ok=True)
    
    # Load coalescent results to know which files succeeded
    print(f"Loading coalescent tree results from {coalescent_results_file}...")
    with open(coalescent_results_file, 'r') as f:
        coalescent_results = json.load(f)
    
    # Convert absolute paths in JSON to paths relative to current data_directory
    # The JSON may contain old absolute paths that need to be converted
    def convert_maf_path(old_path, data_dir):
        """Convert absolute path from JSON to path relative to current data_directory."""
        if not os.path.isabs(old_path):
            # Already a relative path, join with data_directory if needed
            if os.path.isabs(data_dir):
                # If data_dir is absolute, use old_path directly if it exists
                if os.path.exists(old_path):
                    return old_path
                # Otherwise try joining with data_dir
                return os.path.join(data_dir, old_path)
            return old_path
        
        # Extract patient_uuid and filename from old absolute path
        # Pattern: /old/root/data/snv maf/patient_uuid/filename
        old_path_obj = Path(old_path)
        patient_uuid = old_path_obj.parent.name
        filename = old_path_obj.name
        
        # Reconstruct path using current data_directory
        new_path = os.path.join(data_dir, patient_uuid, filename)
        return new_path
    
    # Find MAF files from successful reconstructions and convert paths
    successful_results = []
    for r in coalescent_results:
        if r.get('status') == 'success':
            # Create a copy and update the maf_file path
            result_copy = r.copy()
            result_copy['maf_file'] = convert_maf_path(r['maf_file'], data_directory)
            successful_results.append(result_copy)
    
    print(f"Found {len(successful_results)} successful coalescent reconstructions")
    print(f"Processing spatial mapping for each tree...\n")
    
    results_summary = []
    success_count = 0
    error_count = 0
    
    for idx, result in enumerate(successful_results, 1):
        patient_uuid = result['patient_uuid']
        maf_file = result['maf_file']
        
        print(f"Processing {idx}/{len(successful_results)}: {patient_uuid}")
        
        # Find corresponding coalescent result to get saved tree structure
        coalescent_result = None
        for c_result in coalescent_results:
            if c_result.get('patient_uuid') == patient_uuid and c_result.get('status') == 'success':
                coalescent_result = c_result
                break
        
        result_spatial = process_single_tree_for_spatial_mapping(
            maf_file, spatial_dir, coalescent_result=coalescent_result)
        results_summary.append(result_spatial)
        
        if result_spatial['status'] == 'success':
            spatial_metrics = result_spatial.get('spatial_metrics', {})
            morans_i = spatial_metrics.get('morans_i', 0)
            alpha = spatial_metrics.get('within_lineage_spread_alpha', 0)
            beta = spatial_metrics.get('founder_separation_beta', 0)
            print(f"  ✅ Success: Moran's I={morans_i:.3f}, α={alpha:.2f}, β={beta:.2f}")
            success_count += 1
        else:
            print(f"  ❌ Error: {result_spatial.get('error', 'Unknown error')}")
            error_count += 1
    
    # Calculate summary statistics
    successful_spatial = [r for r in results_summary if r.get('status') == 'success']
    
    alpha_values = []
    beta_values = []
    
    if successful_spatial:
        morans_i_values = [r['spatial_metrics']['morans_i'] for r in successful_spatial]
        alpha_values = [r['spatial_metrics']['within_lineage_spread_alpha'] for r in successful_spatial]
        beta_values = [r['spatial_metrics']['founder_separation_beta'] for r in successful_spatial]
        ratio_values = [r['spatial_metrics']['beta_alpha_ratio'] for r in successful_spatial]
        
        metrics_summary = {
            'mean_morans_i': np.mean(morans_i_values),
            'std_morans_i': np.std(morans_i_values),
            'mean_alpha': np.mean(alpha_values),
            'std_alpha': np.std(alpha_values),
            'mean_beta': np.mean(beta_values),
            'std_beta': np.std(beta_values),
            'mean_ratio': np.mean(ratio_values),
            'std_ratio': np.std(ratio_values)
        }
    else:
        metrics_summary = {}
    
    # Save detailed results
    results_file = os.path.join(spatial_dir, "spatial_mapping_results.json")
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Create summary report
    summary_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_processed': len(successful_results),
        'success_count': success_count,
        'error_count': error_count,
        'metrics_summary': metrics_summary
    }
    create_summary_report(summary_data, spatial_dir, alpha_values=alpha_values, beta_values=beta_values)
    
    print(f"\n{'='*60}")
    print(f"✅ Batch spatial mapping complete!")
    print(f"📊 Results summary:")
    print(f"   • Successful mappings: {success_count}")
    print(f"   • Errors: {error_count}")
    print(f"   • Total processed: {len(successful_results)}")
    print(f"📄 Detailed results saved to: {results_file}")
    print(f"📊 Summary report saved to: {os.path.join(spatial_dir, 'spatial_summary.md')}")
    
    return results_summary


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: batch_spatial_mapping.py <data_directory> <coalescent_results_json> <results_directory>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    coalescent_results = sys.argv[2]
    results_dir = sys.argv[3] if len(sys.argv) > 3 else 'results'
    
    main(data_dir, coalescent_results, results_dir)

