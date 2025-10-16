import json
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Dict, List, Tuple, Optional

# Set Times New Roman as the font for all plots
matplotlib.rcParams['font.family'] = 'Times New Roman'

# Paths to the JSON files
PROCEDURALLY_GENERATED_JSON = "cluster_scripts/cluster_data/metrics_custom-generated.json"
FOR_INSTANCE_JSON = "cluster_scripts/cluster_data/metrics_for-instance.json"

# Class names for better visualization
CLASS_NAMES = {
    0: "Terrain",
    1: "Low Vegetation", 
    2: "Stem",
    3: "Leafy Branches",
    4: "Woody Branches"
}

# Colors for the two methods
PROCEDURALLY_GENERATED_COLOR = '#2E86AB'      # Blue
FOR_INSTANCE_COLOR = '#A23B72'  # Purple


def load_metrics(json_path: str) -> Optional[Dict]:
    """Load metrics from JSON file."""
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found")
        return None
    
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_aggregate_metrics(metrics: Dict) -> Dict:
    """Extract aggregate metrics from the loaded data."""
    aggregate = metrics.get('aggregate', {})
    return {
        'classification_miou': aggregate.get('classification_miou_weighted'),
        'per_class_iou': aggregate.get('per_class_iou_mean', {}),
        'instance_detection_miou': aggregate.get('instance_detection_miou_mean'),
        'instance_detection_f1': aggregate.get('instance_detection_f1_mean'),
        'num_files': metrics.get('total_files', 0),
        'total_points': metrics.get('total_points', 0)
    }


def create_per_class_iou_comparison(procedurally_generated_metrics: Dict, for_instance_metrics: Dict):
    """Create bar chart comparing per-class IoU between methods."""
    fig, ax = plt.subplots(figsize=(12, 5))  # Reduced height for less vertical space
    
    classes = [0, 1, 2, 3, 4]
    class_labels = [CLASS_NAMES[c] for c in classes]
    
    procedurally_generated_ious = []
    for_instance_ious = []
    
    for cls in classes:
        procedurally_generated_iou = procedurally_generated_metrics['per_class_iou'].get(str(cls))
        for_instance_iou = for_instance_metrics['per_class_iou'].get(str(cls))
        
        procedurally_generated_ious.append(procedurally_generated_iou if procedurally_generated_iou is not None else 0)
        for_instance_ious.append(for_instance_iou if for_instance_iou is not None else 0)
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, procedurally_generated_ious, width, label='Synthetic Prediction', 
                   color=PROCEDURALLY_GENERATED_COLOR, alpha=0.8)
    bars2 = ax.bar(x + width/2, for_instance_ious, width, label='For-Instance Prediction', 
                   color=FOR_INSTANCE_COLOR, alpha=0.8)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('IoU')
    ax.set_title('Per-Class IoU Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limit based on max value in the data
    max_value = max(max(procedurally_generated_ious), max(for_instance_ious))
    ax.set_ylim(0, min(1.0, max_value * 1.1))  # 10% padding above max value, but cap at 1.0
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig


def create_overall_metrics_comparison(procedurally_generated_metrics: Dict, for_instance_metrics: Dict):
    """Create bar chart comparing overall metrics between methods."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metrics_names = ['Classification mIoU', 'Instance Detection F1']
    procedurally_generated_values = [
        procedurally_generated_metrics['classification_miou'],
        procedurally_generated_metrics['instance_detection_f1']
    ]
    for_instance_values = [
        for_instance_metrics['classification_miou'],
        for_instance_metrics['instance_detection_f1']
    ]
    
    # Handle None values
    procedurally_generated_values = [v if v is not None else 0 for v in procedurally_generated_values]
    for_instance_values = [v if v is not None else 0 for v in for_instance_values]
    
    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, procedurally_generated_values, width, label='Synthetic Prediction', 
                   color=PROCEDURALLY_GENERATED_COLOR, alpha=0.8)
    bars2 = ax.bar(x + width/2, for_instance_values, width, label='For-Instance Prediction', 
                   color=FOR_INSTANCE_COLOR, alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Overall Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limit based on max value in the data
    max_value = max(max(procedurally_generated_values), max(for_instance_values))
    ax.set_ylim(0, min(1.0, max_value * 1.1))  # 10% padding above max value, but cap at 1.0
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_detailed_comparison_table(procedurally_generated_metrics: Dict, for_instance_metrics: Dict):
    """Create a detailed comparison table."""
    print("\n" + "="*80)
    print("DETAILED METRICS COMPARISON")
    print("="*80)
    
    print(f"\nNumber of files processed:")
    print(f"  Procedurally Generated Prediction: {procedurally_generated_metrics['num_files']} files, {procedurally_generated_metrics['total_points']} points")
    print(f"  For-Instance Prediction: {for_instance_metrics['num_files']} files, {for_instance_metrics['total_points']} points")
    
    print(f"\nOverall Metrics:")
    print(f"  Classification mIoU:")
    print(f"    Procedurally Generated: {procedurally_generated_metrics['classification_miou']:.4f}")
    print(f"    For-Instance: {for_instance_metrics['classification_miou']:.4f}")
    print(f"    Difference: {for_instance_metrics['classification_miou'] - procedurally_generated_metrics['classification_miou']:+.4f}")
    
    print(f"  Instance Detection F1:")
    print(f"    Procedurally Generated: {procedurally_generated_metrics['instance_detection_f1']:.4f}")
    print(f"    For-Instance: {for_instance_metrics['instance_detection_f1']:.4f}")
    print(f"    Difference: {for_instance_metrics['instance_detection_f1'] - procedurally_generated_metrics['instance_detection_f1']:+.4f}")
    
    print(f"\nPer-Class IoU:")
    for cls in [0, 1, 2, 3, 4]:
        procedurally_generated_iou = procedurally_generated_metrics['per_class_iou'].get(str(cls))
        for_instance_iou = for_instance_metrics['per_class_iou'].get(str(cls))
        diff = (for_instance_iou - procedurally_generated_iou) if (procedurally_generated_iou is not None and for_instance_iou is not None) else None
        
        print(f"  {CLASS_NAMES[cls]}:")
        print(f"    Procedurally Generated: {procedurally_generated_iou:.4f}" if procedurally_generated_iou is not None else "    Procedurally Generated: N/A")
        print(f"    For-Instance: {for_instance_iou:.4f}" if for_instance_iou is not None else "    For-Instance: N/A")
        if diff is not None:
            print(f"    Difference: {diff:+.4f}")


def main():
    """Main function to create comparison visualizations."""
    print("Loading metrics data...")
    
    # Load metrics
    procedurally_generated_data = load_metrics(PROCEDURALLY_GENERATED_JSON)
    for_instance_data = load_metrics(FOR_INSTANCE_JSON)
    
    if procedurally_generated_data is None or for_instance_data is None:
        print("Error: Could not load one or both metrics files")
        return
    
    # Extract aggregate metrics
    procedurally_generated_metrics = extract_aggregate_metrics(procedurally_generated_data)
    for_instance_metrics = extract_aggregate_metrics(for_instance_data)
    
    # Print detailed comparison
    create_detailed_comparison_table(procedurally_generated_metrics, for_instance_metrics)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Per-class IoU comparison
    fig1 = create_per_class_iou_comparison(procedurally_generated_metrics, for_instance_metrics)
    fig1.savefig('cluster_scripts/cluster_data/per_class_iou_comparison.pdf', 
                 dpi=300, bbox_inches='tight')
    print("Saved: per_class_iou_comparison.pdf")
    
    # Overall metrics comparison
    fig2 = create_overall_metrics_comparison(procedurally_generated_metrics, for_instance_metrics)
    fig2.savefig('cluster_scripts/cluster_data/overall_metrics_comparison.pdf', 
                 dpi=300, bbox_inches='tight')
    print("Saved: overall_metrics_comparison.pdf")
    
    # Show plots
    plt.show()
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
