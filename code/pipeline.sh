#!/bin/bash

# MAF Analysis Pipeline
# This script runs the complete workflow for analyzing SNV MAF data

set -e  # Exit on any error

# Configuration
PROJECT_ROOT="/Users/jeongwoohyun/thesis"
CODE_DIR="$PROJECT_ROOT/code"
RESULTS_DIR="$PROJECT_ROOT/results"
DATA_DIR="$PROJECT_ROOT/data/snv maf"

# Function to print output
print_status() {
    echo "[INFO] $1"
}

print_success() {
    echo "[SUCCESS] $1"
}

print_warning() {
    echo "[WARNING] $1"
}

print_error() {
    echo "[ERROR] $1"
}

# Function to check if a file exists
check_file() {
    if [ ! -f "$1" ]; then
        print_error "File not found: $1"
        exit 1
    fi
}

# Function to check if a directory exists
check_directory() {
    if [ ! -d "$1" ]; then
        print_error "Directory not found: $1"
        exit 1
    fi
}

# Main pipeline function
run_pipeline() {
    print_status "Starting MAF Analysis Pipeline"
    echo "=========================================="
    
    # Check required directories and files
    check_directory "$PROJECT_ROOT"
    check_directory "$CODE_DIR"
    check_directory "$DATA_DIR"
    
    # Create results directory if it doesn't exist
    mkdir -p "$RESULTS_DIR"
    
    # Step 1: Preprocessing
    print_status "Step 1: Preprocessing MAF files"
    echo "----------------------------------------"
    check_file "$CODE_DIR/preprocessing.py"
    
    cd "$PROJECT_ROOT"
    python "$CODE_DIR/preprocessing.py"
    
    if [ $? -eq 0 ]; then
        print_success "Preprocessing completed successfully"
        print_status "Results saved to: $RESULTS_DIR/preprocessing_results.md"
    else
        print_error "Preprocessing failed"
        exit 1
    fi
    
    echo ""
    
    # Step 2: Neutrality Testing
    print_status "Step 2: Neutrality Testing"
    echo "----------------------------------------"
    check_file "$CODE_DIR/batch_neutrality_test.py"
    
    cd "$PROJECT_ROOT"
    python "$CODE_DIR/batch_neutrality_test.py" "$DATA_DIR" "$RESULTS_DIR"
    
    if [ $? -eq 0 ]; then
        print_success "Neutrality testing completed successfully"
        print_status "Results saved to: $RESULTS_DIR/neutrality_tests/"
    else
        print_error "Neutrality testing failed"
        exit 1
    fi
    
    echo ""
    
    # Step 3: Coalescent Tree Reconstruction
    print_status "Step 3: Coalescent Tree Reconstruction"
    echo "----------------------------------------"
    check_file "$CODE_DIR/batch_coalescent_reconstruction.py"
    
    cd "$PROJECT_ROOT"
    python "$CODE_DIR/batch_coalescent_reconstruction.py" "$DATA_DIR" "$RESULTS_DIR"
    
    if [ $? -eq 0 ]; then
        print_success "Coalescent tree reconstruction completed successfully"
        print_status "Results saved to: $RESULTS_DIR/coalescent_trees/"
    else
        print_error "Coalescent tree reconstruction failed"
        exit 1
    fi
    
    echo ""
    
    # Step 4: Spatial Mapping
    print_status "Step 4: Spatial Mapping"
    echo "----------------------------------------"
    check_file "$CODE_DIR/batch_spatial_mapping.py"
    
    COALESCENT_RESULTS="$RESULTS_DIR/coalescent_trees/coalescent_tree_results.json"
    if [ ! -f "$COALESCENT_RESULTS" ]; then
        print_error "Coalescent tree results file not found: $COALESCENT_RESULTS"
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
    python "$CODE_DIR/batch_spatial_mapping.py" "$DATA_DIR" "$COALESCENT_RESULTS" "$RESULTS_DIR"
    
    if [ $? -eq 0 ]; then
        print_success "Spatial mapping completed successfully"
        print_status "Results saved to: $RESULTS_DIR/spatial_mapping/"
    else
        print_error "Spatial mapping failed"
        exit 1
    fi
    
    echo ""
    print_status "Pipeline completed successfully!"
    echo "=========================================="
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -p, --preprocess-only  Run only preprocessing step"
    echo "  -n, --neutrality-only Run only neutrality testing step"
    echo "  -c, --coalescent-only Run only coalescent tree reconstruction step"
    echo "  -s, --spatial-only Run only spatial mapping step"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run complete pipeline"
    echo "  $0 --preprocess-only  # Run only preprocessing"
    echo "  $0 --neutrality-only # Run only neutrality testing"
    echo "  $0 --coalescent-only # Run only coalescent tree reconstruction"
    echo "  $0 --spatial-only # Run only spatial mapping"
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_usage
        exit 0
        ;;
    -p|--preprocess-only)
        print_status "Running preprocessing only"
        check_directory "$PROJECT_ROOT"
        check_directory "$CODE_DIR"
        check_directory "$DATA_DIR"
        mkdir -p "$RESULTS_DIR"
        
        check_file "$CODE_DIR/preprocessing.py"
        cd "$PROJECT_ROOT"
        python "$CODE_DIR/preprocessing.py"
        
        if [ $? -eq 0 ]; then
            print_success "Preprocessing completed successfully"
        else
            print_error "Preprocessing failed"
            exit 1
        fi
        exit 0
        ;;
    -n|--neutrality-only)
        print_status "Running neutrality testing only"
        check_directory "$PROJECT_ROOT"
        check_directory "$CODE_DIR"
        check_directory "$DATA_DIR"
        mkdir -p "$RESULTS_DIR"
        
        check_file "$CODE_DIR/batch_neutrality_test.py"
        cd "$PROJECT_ROOT"
        python "$CODE_DIR/batch_neutrality_test.py" "$DATA_DIR" "$RESULTS_DIR"
        
        if [ $? -eq 0 ]; then
            print_success "Neutrality testing completed successfully"
        else
            print_error "Neutrality testing failed"
            exit 1
        fi
        exit 0
        ;;
    -c|--coalescent-only)
        print_status "Running coalescent tree reconstruction only"
        check_directory "$PROJECT_ROOT"
        check_directory "$CODE_DIR"
        check_directory "$DATA_DIR"
        mkdir -p "$RESULTS_DIR"
        
        check_file "$CODE_DIR/batch_coalescent_reconstruction.py"
        cd "$PROJECT_ROOT"
        python "$CODE_DIR/batch_coalescent_reconstruction.py" "$DATA_DIR" "$RESULTS_DIR"
        
        if [ $? -eq 0 ]; then
            print_success "Coalescent tree reconstruction completed successfully"
        else
            print_error "Coalescent tree reconstruction failed"
            exit 1
        fi
        exit 0
        ;;
    -s|--spatial-only)
        print_status "Running spatial mapping only"
        check_directory "$PROJECT_ROOT"
        check_directory "$CODE_DIR"
        check_directory "$DATA_DIR"
        mkdir -p "$RESULTS_DIR"
        
        check_file "$CODE_DIR/batch_spatial_mapping.py"
        
        COALESCENT_RESULTS="$RESULTS_DIR/coalescent_trees/coalescent_tree_results.json"
        if [ ! -f "$COALESCENT_RESULTS" ]; then
            print_error "Coalescent tree results file not found: $COALESCENT_RESULTS"
            print_error "Please run coalescent tree reconstruction first"
            exit 1
        fi
        
        cd "$PROJECT_ROOT"
        python "$CODE_DIR/batch_spatial_mapping.py" "$DATA_DIR" "$COALESCENT_RESULTS" "$RESULTS_DIR"
        
        if [ $? -eq 0 ]; then
            print_success "Spatial mapping completed successfully"
        else
            print_error "Spatial mapping failed"
            exit 1
        fi
        exit 0
        ;;
    "")
        run_pipeline
        ;;
    *)
        print_error "Unknown option: $1"
        show_usage
        exit 1
        ;;
esac
