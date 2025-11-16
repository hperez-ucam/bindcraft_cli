#!/usr/bin/env python3
"""
BindCraft CLI: Protein binder design pipeline
Executes the BindCraft design pipeline from command line using a configuration file.
"""

import os
import sys
import json
import time
import argparse
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import threading

# Add bindcraft directory to Python path if it exists
SCRIPT_DIR = Path(__file__).parent.resolve()
BINDCRAFT_DIR = SCRIPT_DIR / "bindcraft"
if BINDCRAFT_DIR.exists():
    sys.path.insert(0, str(SCRIPT_DIR))

# Import BindCraft functions - these will be available after installation
# Note: This import may fail if dependencies (like pandas) have issues
# We'll check later in main() if the functions are actually available
try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from bindcraft.functions import *
except (ImportError, AttributeError, ModuleNotFoundError) as e:
    # Store the error for later reporting
    _bindcraft_import_error = e
    pass

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_timestamp():
    """Get formatted timestamp for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_progress(message, level="INFO"):
    """Log progress message with timestamp."""
    timestamp = get_timestamp()
    print(f"[{timestamp}] [{level}] {message}")
    sys.stdout.flush()  # Force flush to ensure immediate output

def validate_config(config):
    """Validate configuration parameters."""
    required_fields = [
        "design_path", "binder_name", "starting_pdb", "chains",
        "lengths", "number_of_final_designs", "design_protocol",
        "prediction_protocol", "interface_protocol", "template_protocol",
        "filter_option"
    ]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate lengths
    if isinstance(config["lengths"], list):
        if len(config["lengths"]) != 2:
            raise ValueError("lengths must be a list of two integers [min, max]")
        config["lengths"] = config["lengths"]
    elif isinstance(config["lengths"], str):
        lengths = [int(x.strip()) for x in config["lengths"].split(',')]
        if len(lengths) != 2:
            raise ValueError("lengths must be two comma-separated integers")
        config["lengths"] = lengths
    else:
        raise ValueError("lengths must be a list or comma-separated string")
    
    return config

def setup_paths(config):
    """Setup and create necessary directories."""
    design_path = Path(config["design_path"]).expanduser().resolve()
    design_path.mkdir(parents=True, exist_ok=True)
    config["design_path"] = str(design_path)
    return config

def generate_advanced_settings_path(config):
    """Generate path to advanced settings JSON based on protocol choices."""
    design_protocol = config["design_protocol"]
    interface_protocol = config["interface_protocol"]
    template_protocol = config["template_protocol"]
    prediction_protocol = config["prediction_protocol"]
    
    # Map design protocols
    if design_protocol == "Default":
        design_protocol_tag = "default_4stage_multimer"
    elif design_protocol == "Beta-sheet":
        design_protocol_tag = "betasheet_4stage_multimer"
    elif design_protocol == "Peptide":
        design_protocol_tag = "peptide_3stage_multimer"
    else:
        raise ValueError(f"Unsupported design protocol: {design_protocol}")
    
    # Map interface protocols
    if interface_protocol == "AlphaFold2":
        interface_protocol_tag = ""
    elif interface_protocol == "MPNN":
        interface_protocol_tag = "_mpnn"
    else:
        raise ValueError(f"Unsupported interface protocol: {interface_protocol}")
    
    # Map template protocols
    if template_protocol == "Default":
        template_protocol_tag = ""
    elif template_protocol == "Masked":
        template_protocol_tag = "_flexible"
    else:
        raise ValueError(f"Unsupported template protocol: {template_protocol}")
    
    # Map prediction protocols
    if design_protocol == "Peptide":
        prediction_protocol_tag = ""
    else:
        if prediction_protocol == "Default":
            prediction_protocol_tag = ""
        elif prediction_protocol == "HardTarget":
            prediction_protocol_tag = "_hardtarget"
        else:
            raise ValueError(f"Unsupported prediction protocol: {prediction_protocol}")
    
    advanced_settings_path = f"bindcraft/settings_advanced/{design_protocol_tag}{interface_protocol_tag}{template_protocol_tag}{prediction_protocol_tag}.json"
    return advanced_settings_path

def generate_filter_settings_path(config):
    """Generate path to filter settings JSON."""
    filter_option = config["filter_option"]
    
    filter_map = {
        "Default": "bindcraft/settings_filters/default_filters.json",
        "Peptide": "bindcraft/settings_filters/peptide_filters.json",
        "Relaxed": "bindcraft/settings_filters/relaxed_filters.json",
        "Peptide_Relaxed": "bindcraft/settings_filters/peptide_relaxed_filters.json",
        "None": "bindcraft/settings_filters/no_filters.json"
    }
    
    if filter_option not in filter_map:
        raise ValueError(f"Unsupported filter option: {filter_option}")
    
    return filter_map[filter_option]

def save_target_settings(config):
    """Save target settings to JSON file."""
    if config.get("load_previous_target_settings"):
        target_settings_path = config["load_previous_target_settings"]
        print(f"Loading previous target settings from: {target_settings_path}")
        return target_settings_path
    
    settings = {
        "design_path": config["design_path"],
        "binder_name": config["binder_name"],
        "starting_pdb": config["starting_pdb"],
        "chains": config["chains"],
        "target_hotspot_residues": config.get("target_hotspot_residues", ""),
        "lengths": config["lengths"],
        "number_of_final_designs": config["number_of_final_designs"]
    }
    
    target_settings_path = os.path.join(config["design_path"], config["binder_name"] + ".json")
    os.makedirs(config["design_path"], exist_ok=True)
    
    with open(target_settings_path, 'w') as f:
        json.dump(settings, f, indent=4)
    
    print(f"Target settings saved to: {target_settings_path}")
    return target_settings_path

def main():
    parser = argparse.ArgumentParser(
        description="BindCraft: Protein binder design pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python bindcraft_cli.py --config config.json
  
Configuration file should contain:
  - design_path: Output directory for designs
  - binder_name: Prefix for binder names
  - starting_pdb: Path to target PDB file
  - chains: Target chains (e.g., "A" or "A,C")
  - target_hotspot_residues: Specific residues to target (optional)
  - lengths: Binder length range [min, max]
  - number_of_final_designs: Target number of final designs
  - design_protocol: "Default", "Beta-sheet", or "Peptide"
  - prediction_protocol: "Default" or "HardTarget"
  - interface_protocol: "AlphaFold2" or "MPNN"
  - template_protocol: "Default" or "Masked"
  - filter_option: "Default", "Peptide", "Relaxed", "Peptide_Relaxed", or "None"
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file"
    )
    
    args = parser.parse_args()
    
    # Load and validate configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    config = validate_config(config)
    config = setup_paths(config)
    
    # Generate settings paths
    target_settings_path = save_target_settings(config)
    advanced_settings_path = generate_advanced_settings_path(config)
    filter_settings_path = generate_filter_settings_path(config)
    
    print(f"\n{'='*60}")
    print("BindCraft Configuration")
    print(f"{'='*60}")
    print(f"Design Path: {config['design_path']}")
    print(f"Binder Name: {config['binder_name']}")
    print(f"Starting PDB: {config['starting_pdb']}")
    print(f"Chains: {config['chains']}")
    print(f"Length Range: {config['lengths']}")
    print(f"Target Designs: {config['number_of_final_designs']}")
    print(f"Design Protocol: {config['design_protocol']}")
    print(f"Prediction Protocol: {config['prediction_protocol']}")
    print(f"Interface Protocol: {config['interface_protocol']}")
    print(f"Template Protocol: {config['template_protocol']}")
    print(f"Filter Option: {config['filter_option']}")
    print(f"\nTarget Settings: {target_settings_path}")
    print(f"Advanced Settings: {advanced_settings_path}")
    print(f"Filter Settings: {filter_settings_path}")
    print(f"{'='*60}\n")
    
    # Check if bindcraft module is available
    try:
        # Try to use a function to verify import worked
        check_jax_gpu  # This will raise NameError if import failed
        print("✓ BindCraft functions imported successfully")
    except NameError:
        print("✗ Error: BindCraft functions not available")
        print("\nPossible causes:")
        print("1. BindCraft is not installed - Run ./install_bindcraft.sh")
        print("2. NumPy/pandas compatibility issue - Try: pip install 'numpy<2'")
        if '_bindcraft_import_error' in globals():
            print(f"\nImport error details: {_bindcraft_import_error}")
        sys.exit(1)
    
    # Prepare arguments for the design pipeline
    pipeline_args = {
        "settings": target_settings_path,
        "filters": filter_settings_path,
        "advanced": advanced_settings_path
    }
    
    # Check if JAX-capable GPU is available
    try:
        check_jax_gpu()
        print("✓ GPU check passed")
    except Exception as e:
        print(f"✗ GPU check failed: {e}")
        print("Continuing anyway...")
    
    # Load settings from JSON
    print("\nLoading settings from JSON files...")
    settings_path, filters_path, advanced_path = (
        pipeline_args["settings"],
        pipeline_args["filters"],
        pipeline_args["advanced"]
    )
    
    target_settings, advanced_settings, filters = load_json_settings(
        settings_path, filters_path, advanced_path
    )
    
    settings_file = os.path.basename(settings_path).split('.')[0]
    filters_file = os.path.basename(filters_path).split('.')[0]
    advanced_file = os.path.basename(advanced_path).split('.')[0]
    
    # Load AF2 model settings
    print("Loading AlphaFold2 model settings...")
    design_models, prediction_models, multimer_validation = load_af2_models(
        advanced_settings["use_multimer_design"]
    )
    
    # Perform checks on advanced_settings
    bindcraft_folder = "bindcraft"  # Changed from "colab" for CLI
    advanced_settings = perform_advanced_settings_check(advanced_settings, bindcraft_folder)
    
    # Generate directories
    print("Generating output directories...")
    design_paths = generate_directories(target_settings["design_path"])
    
    # Generate dataframes
    print("Initializing dataframes...")
    trajectory_labels, design_labels, final_labels = generate_dataframe_labels()
    
    trajectory_csv = os.path.join(target_settings["design_path"], 'trajectory_stats.csv')
    mpnn_csv = os.path.join(target_settings["design_path"], 'mpnn_design_stats.csv')
    final_csv = os.path.join(target_settings["design_path"], 'final_design_stats.csv')
    failure_csv = os.path.join(target_settings["design_path"], 'failure_csv.csv')
    
    create_dataframe(trajectory_csv, trajectory_labels)
    create_dataframe(mpnn_csv, design_labels)
    create_dataframe(final_csv, final_labels)
    generate_filter_pass_csv(failure_csv, pipeline_args["filters"])
    
    # Initialize PyRosetta
    print("\nInitializing PyRosetta...")
    pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')
    print("✓ PyRosetta initialized")
    
    # Run BindCraft pipeline
    print(f"\n{'='*60}")
    print("Starting BindCraft Design Pipeline")
    print(f"{'='*60}\n")
    
    # Initialize counters
    script_start_time = time.time()
    trajectory_n = 1
    accepted_designs = 0
    
    # Start design loop
    while True:
        # Check if we have the target number of binders
        final_designs_reached = check_accepted_designs(
            design_paths, mpnn_csv, final_labels, final_csv,
            advanced_settings, target_settings, design_labels
        )
        
        if final_designs_reached:
            print(f"\n✓ Target number of designs ({target_settings['number_of_final_designs']}) reached!")
            break
        
        # Check if we reached maximum allowed trajectories
        max_trajectories_reached = check_n_trajectories(design_paths, advanced_settings)
        
        if max_trajectories_reached:
            print("\n⚠ Maximum trajectories reached")
            break
        
        # Initialize design
        trajectory_start_time = time.time()
        
        # Generate random seed
        seed = int(np.random.randint(0, high=999999, size=1, dtype=int)[0])
        
        # Sample binder design length
        samples = np.arange(min(target_settings["lengths"]), max(target_settings["lengths"]) + 1)
        length = np.random.choice(samples)
        
        # Load desired helicity value
        helicity_value = load_helicity(advanced_settings)
        
        # Generate design name and check if same trajectory was already run
        design_name = target_settings["binder_name"] + "_l" + str(length) + "_s" + str(seed)
        trajectory_dirs = ["Trajectory", "Trajectory/Relaxed", "Trajectory/LowConfidence", "Trajectory/Clashing"]
        trajectory_exists = any(
            os.path.exists(os.path.join(design_paths[trajectory_dir], design_name + ".pdb"))
            for trajectory_dir in trajectory_dirs
        )
        
        if not trajectory_exists:
            log_progress(f"[{trajectory_n}] Starting trajectory: {design_name}", "TRAJECTORY")
            log_progress(f"  Length: {length}, Seed: {seed}, Helicity: {helicity_value}", "INFO")
            
            # Begin binder hallucination
            log_progress("  Stage 1: Starting binder hallucination (this may take 5-30 minutes)...", "STAGE")
            log_progress("  Stage 1: Optimizing binder structure using AlphaFold2...", "PROGRESS")
            log_progress("  Stage 1: This is a long-running process. The script is working, please wait...", "INFO")
            hallucination_start = time.time()
            
            trajectory = binder_hallucination(
                design_name, target_settings["starting_pdb"], target_settings["chains"],
                target_settings["target_hotspot_residues"], length, seed, helicity_value,
                design_models, advanced_settings, design_paths, failure_csv
            )
            
            hallucination_time = time.time() - hallucination_start
            log_progress(f"  Stage 1: ✓ Binder hallucination completed in {int(hallucination_time//60)}m {int(hallucination_time%60)}s", "STAGE")
            trajectory_metrics = copy_dict(trajectory._tmp["best"]["aux"]["log"])
            trajectory_pdb = os.path.join(design_paths["Trajectory"], design_name + ".pdb")
            
            # Round the metrics
            trajectory_metrics = {k: round(v, 2) if isinstance(v, float) else v for k, v in trajectory_metrics.items()}
            
            # Time trajectory
            trajectory_time = time.time() - trajectory_start_time
            trajectory_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(trajectory_time // 3600), int((trajectory_time % 3600) // 60), int(trajectory_time % 60))}"
            log_progress(f"  Trajectory completed in: {trajectory_time_text}", "TRAJECTORY")
            
            # Proceed if there is no trajectory termination signal
            if trajectory.aux["log"]["terminate"] == "":
                log_progress("  Stage 2: Relaxing trajectory structure with PyRosetta...", "STAGE")
                # Relax binder to calculate statistics
                trajectory_relaxed = os.path.join(design_paths["Trajectory/Relaxed"], design_name + ".pdb")
                pr_relax(trajectory_pdb, trajectory_relaxed)
                log_progress("  Stage 2: Relaxation completed", "STAGE")
                
                log_progress("  Stage 3: Calculating trajectory statistics...", "STAGE")
                
                # Define binder chain
                binder_chain = "B"
                
                # Calculate clashes
                num_clashes_trajectory = calculate_clash_score(trajectory_pdb)
                num_clashes_relaxed = calculate_clash_score(trajectory_relaxed)
                
                # Secondary structure content
                trajectory_alpha, trajectory_beta, trajectory_loops, trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, trajectory_i_plddt, trajectory_ss_plddt = calc_ss_percentage(
                    trajectory_pdb, advanced_settings, binder_chain
                )
                
                # Analyze interface scores
                trajectory_interface_scores, trajectory_interface_AA, trajectory_interface_residues = score_interface(
                    trajectory_relaxed, binder_chain
                )
                
                # Starting binder sequence
                trajectory_sequence = trajectory.get_seq(get_best=True)[0]
                
                # Analyze sequence
                traj_seq_notes = validate_design_sequence(
                    trajectory_sequence, num_clashes_relaxed, advanced_settings
                )
                
                # Target structure RMSD
                trajectory_target_rmsd = unaligned_rmsd(
                    target_settings["starting_pdb"], trajectory_pdb,
                    target_settings["chains"], 'A'
                )
                
                # Save trajectory statistics
                log_progress(f"  Stage 3: Statistics - pLDDT: {trajectory_metrics['plddt']:.2f}, iPTM: {trajectory_metrics['i_ptm']:.2f}, Interface residues: {trajectory_interface_residues}", "INFO")
                trajectory_data = [
                    design_name, advanced_settings["design_algorithm"], length, seed, helicity_value,
                    target_settings["target_hotspot_residues"], trajectory_sequence, trajectory_interface_residues,
                    trajectory_metrics['plddt'], trajectory_metrics['ptm'], trajectory_metrics['i_ptm'],
                    trajectory_metrics['pae'], trajectory_metrics['i_pae'],
                    trajectory_i_plddt, trajectory_ss_plddt, num_clashes_trajectory, num_clashes_relaxed,
                    trajectory_interface_scores['binder_score'],
                    trajectory_interface_scores['surface_hydrophobicity'],
                    trajectory_interface_scores['interface_sc'],
                    trajectory_interface_scores['interface_packstat'],
                    trajectory_interface_scores['interface_dG'],
                    trajectory_interface_scores['interface_dSASA'],
                    trajectory_interface_scores['interface_dG_SASA_ratio'],
                    trajectory_interface_scores['interface_fraction'],
                    trajectory_interface_scores['interface_hydrophobicity'],
                    trajectory_interface_scores['interface_nres'],
                    trajectory_interface_scores['interface_interface_hbonds'],
                    trajectory_interface_scores['interface_hbond_percentage'],
                    trajectory_interface_scores['interface_delta_unsat_hbonds'],
                    trajectory_interface_scores['interface_delta_unsat_hbonds_percentage'],
                    trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface,
                    trajectory_alpha, trajectory_beta, trajectory_loops, trajectory_interface_AA,
                    trajectory_target_rmsd, trajectory_time_text, traj_seq_notes,
                    settings_file, filters_file, advanced_file
                ]
                insert_data(trajectory_csv, trajectory_data)
                log_progress("  Stage 3: Statistics saved", "STAGE")
                
                if advanced_settings["enable_mpnn"]:
                    log_progress("  Stage 4: Starting MPNN sequence redesign...", "STAGE")
                    # initialise MPNN counters
                    mpnn_n = 1
                    accepted_mpnn = 0
                    mpnn_dict = {}
                    design_start_time = time.time()

                    ### MPNN redesign of starting binder
                    log_progress("  Stage 4: Generating MPNN sequences...", "PROGRESS")
                    mpnn_trajectories = mpnn_gen_sequence(trajectory_pdb, binder_chain, trajectory_interface_residues, advanced_settings)
                    log_progress(f"  Stage 4: Generated {advanced_settings['num_seqs']} MPNN sequences", "PROGRESS")
                    existing_mpnn_sequences = set(pd.read_csv(mpnn_csv, usecols=['Sequence'])['Sequence'].values)

                    # create set of MPNN sequences with allowed amino acid composition
                    restricted_AAs = set(aa.strip().upper() for aa in advanced_settings["omit_AAs"].split(',')) if advanced_settings["force_reject_AA"] else set()

                    mpnn_sequences = sorted({
                        mpnn_trajectories['seq'][n][-length:]: {
                            'seq': mpnn_trajectories['seq'][n][-length:],
                            'score': mpnn_trajectories['score'][n],
                            'seqid': mpnn_trajectories['seqid'][n]
                        } for n in range(advanced_settings["num_seqs"])
                        if (not restricted_AAs or not any(aa in mpnn_trajectories['seq'][n][-length:].upper() for aa in restricted_AAs))
                        and mpnn_trajectories['seq'][n][-length:] not in existing_mpnn_sequences
                    }.values(), key=lambda x: x['score'])

                    del existing_mpnn_sequences

                    # check whether any sequences are left after amino acid rejection and duplication check, and if yes proceed with prediction
                    if mpnn_sequences:
                        log_progress(f"  Stage 4: {len(mpnn_sequences)} MPNN sequences to evaluate", "PROGRESS")
                        # add optimisation for increasing recycles if trajectory is beta sheeted
                        if advanced_settings["optimise_beta"] and float(trajectory_beta) > 15:
                            advanced_settings["num_recycles_validation"] = advanced_settings["optimise_beta_recycles_valid"]
                            log_progress(f"  Stage 4: Beta sheet detected ({trajectory_beta:.1f}%), using optimized recycles", "INFO")

                        ### Compile prediction models once for faster prediction of MPNN sequences
                        log_progress("  Stage 4: Compiling AlphaFold2 prediction models (this may take 1-2 minutes)...", "PROGRESS")
                        clear_mem()
                        # compile complex prediction model
                        complex_prediction_model = mk_afdesign_model(protocol="binder", num_recycles=advanced_settings["num_recycles_validation"], data_dir=advanced_settings["af_params_dir"],
                                                                    use_multimer=multimer_validation)
                        complex_prediction_model.prep_inputs(pdb_filename=target_settings["starting_pdb"], chain=target_settings["chains"], binder_len=length, rm_target_seq=advanced_settings["rm_template_seq_predict"],
                                                            rm_target_sc=advanced_settings["rm_template_sc_predict"])
                        log_progress("  Stage 4: Complex prediction model compiled", "PROGRESS")

                        # compile binder monomer prediction model
                        binder_prediction_model = mk_afdesign_model(protocol="hallucination", use_templates=False, initial_guess=False,
                                                                    use_initial_atom_pos=False, num_recycles=advanced_settings["num_recycles_validation"],
                                                                    data_dir=advanced_settings["af_params_dir"], use_multimer=multimer_validation)
                        binder_prediction_model.prep_inputs(length=length)
                        log_progress("  Stage 4: Binder prediction model compiled", "PROGRESS")
                        log_progress("  Stage 4: Starting evaluation of MPNN sequences (this may take 20-60 minutes)...", "PROGRESS")

                        # iterate over designed sequences
                        total_mpnn = len(mpnn_sequences)
                        for mpnn_idx, mpnn_sequence in enumerate(mpnn_sequences, 1):
                            mpnn_time = time.time()

                            # generate mpnn design name numbering
                            mpnn_design_name = design_name + "_mpnn" + str(mpnn_n)
                            mpnn_score = round(mpnn_sequence['score'],2)
                            mpnn_seqid = round(mpnn_sequence['seqid'],2)
                            
                            log_progress(f"  Stage 4: Evaluating MPNN {mpnn_idx}/{total_mpnn}: {mpnn_design_name} (score: {mpnn_score:.2f}, seqid: {mpnn_seqid:.2f})", "MPNN")

                            # add design to dictionary
                            mpnn_dict[mpnn_design_name] = {'seq': mpnn_sequence['seq'], 'score': mpnn_score, 'seqid': mpnn_seqid}

                            # save fasta sequence
                            if advanced_settings["save_mpnn_fasta"] is True:
                                save_fasta(mpnn_design_name, mpnn_sequence['seq'], design_paths)

                            ### Predict mpnn redesigned binder complex using masked templates
                            log_progress(f"    Predicting complex structure for {mpnn_design_name}...", "MPNN")
                            mpnn_complex_statistics, pass_af2_filters = predict_binder_complex(complex_prediction_model,
                                                                                            mpnn_sequence['seq'], mpnn_design_name,
                                                                                            target_settings["starting_pdb"], target_settings["chains"],
                                                                                            length, trajectory_pdb, prediction_models, advanced_settings,
                                                                                            filters, design_paths, failure_csv)

                            # if AF2 filters are not passed then skip the scoring
                            if not pass_af2_filters:
                                log_progress(f"    ✗ Base AF2 filters not passed for {mpnn_design_name}, skipping interface scoring", "MPNN")
                                mpnn_n += 1
                                # Continue to next MPNN sequence
                                continue
                            
                            log_progress(f"    ✓ AF2 filters passed for {mpnn_design_name}, proceeding with scoring...", "MPNN")
                            log_progress(f"    Calculating statistics for {mpnn_design_name}...", "MPNN")

                            # calculate statistics for each model individually
                            for model_num in prediction_models:
                                log_progress(f"      Processing model {model_num+1}/5 for {mpnn_design_name}...", "MPNN")
                                mpnn_design_pdb = os.path.join(design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
                                mpnn_design_relaxed = os.path.join(design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{model_num+1}.pdb")

                                if os.path.exists(mpnn_design_pdb):
                                    # Calculate clashes before and after relaxation
                                    num_clashes_mpnn = calculate_clash_score(mpnn_design_pdb)
                                    num_clashes_mpnn_relaxed = calculate_clash_score(mpnn_design_relaxed)

                                    # analyze interface scores for relaxed af2 trajectory
                                    mpnn_interface_scores, mpnn_interface_AA, mpnn_interface_residues = score_interface(mpnn_design_relaxed, binder_chain)

                                    # secondary structure content of starting trajectory binder
                                    mpnn_alpha, mpnn_beta, mpnn_loops, mpnn_alpha_interface, mpnn_beta_interface, mpnn_loops_interface, mpnn_i_plddt, mpnn_ss_plddt = calc_ss_percentage(mpnn_design_pdb, advanced_settings, binder_chain)

                                    # unaligned RMSD calculate to determine if binder is in the designed binding site
                                    rmsd_site = unaligned_rmsd(trajectory_pdb, mpnn_design_pdb, binder_chain, binder_chain)

                                    # calculate RMSD of target compared to input PDB
                                    target_rmsd = target_pdb_rmsd(mpnn_design_pdb, target_settings["starting_pdb"], target_settings["chains"])

                                    # add the additional statistics to the mpnn_complex_statistics dictionary
                                    mpnn_complex_statistics[model_num+1].update({
                                        'i_pLDDT': mpnn_i_plddt,
                                        'ss_pLDDT': mpnn_ss_plddt,
                                        'Unrelaxed_Clashes': num_clashes_mpnn,
                                        'Relaxed_Clashes': num_clashes_mpnn_relaxed,
                                        'Binder_Energy_Score': mpnn_interface_scores['binder_score'],
                                        'Surface_Hydrophobicity': mpnn_interface_scores['surface_hydrophobicity'],
                                        'ShapeComplementarity': mpnn_interface_scores['interface_sc'],
                                        'PackStat': mpnn_interface_scores['interface_packstat'],
                                        'dG': mpnn_interface_scores['interface_dG'],
                                        'dSASA': mpnn_interface_scores['interface_dSASA'],
                                        'dG/dSASA': mpnn_interface_scores['interface_dG_SASA_ratio'],
                                        'Interface_SASA_%': mpnn_interface_scores['interface_fraction'],
                                        'Interface_Hydrophobicity': mpnn_interface_scores['interface_hydrophobicity'],
                                        'n_InterfaceResidues': mpnn_interface_scores['interface_nres'],
                                        'n_InterfaceHbonds': mpnn_interface_scores['interface_interface_hbonds'],
                                        'InterfaceHbondsPercentage': mpnn_interface_scores['interface_hbond_percentage'],
                                        'n_InterfaceUnsatHbonds': mpnn_interface_scores['interface_delta_unsat_hbonds'],
                                        'InterfaceUnsatHbondsPercentage': mpnn_interface_scores['interface_delta_unsat_hbonds_percentage'],
                                        'InterfaceAAs': mpnn_interface_AA,
                                        'Interface_Helix%': mpnn_alpha_interface,
                                        'Interface_BetaSheet%': mpnn_beta_interface,
                                        'Interface_Loop%': mpnn_loops_interface,
                                        'Binder_Helix%': mpnn_alpha,
                                        'Binder_BetaSheet%': mpnn_beta,
                                        'Binder_Loop%': mpnn_loops,
                                        'Hotspot_RMSD': rmsd_site,
                                        'Target_RMSD': target_rmsd
                                    })

                                    # save space by removing unrelaxed predicted mpnn complex pdb?
                                    if advanced_settings["remove_unrelaxed_complex"]:
                                        os.remove(mpnn_design_pdb)

                            # calculate complex averages
                            mpnn_complex_averages = calculate_averages(mpnn_complex_statistics, handle_aa=True)
                            log_progress(f"    Complex statistics calculated for {mpnn_design_name}", "MPNN")

                            ### Predict binder alone in single sequence mode
                            log_progress(f"    Predicting binder alone structure for {mpnn_design_name}...", "MPNN")
                            binder_statistics = predict_binder_alone(binder_prediction_model, mpnn_sequence['seq'], mpnn_design_name, length,
                                                                    trajectory_pdb, binder_chain, prediction_models, advanced_settings, design_paths)
                            log_progress(f"    Binder prediction completed for {mpnn_design_name}", "MPNN")

                            # extract RMSDs of binder to the original trajectory
                            for model_num in prediction_models:
                                mpnn_binder_pdb = os.path.join(design_paths["MPNN/Binder"], f"{mpnn_design_name}_model{model_num+1}.pdb")

                                if os.path.exists(mpnn_binder_pdb):
                                    rmsd_binder = unaligned_rmsd(trajectory_pdb, mpnn_binder_pdb, binder_chain, "A")

                                # append to statistics
                                binder_statistics[model_num+1].update({
                                        'Binder_RMSD': rmsd_binder
                                    })

                                # save space by removing binder monomer models?
                                if advanced_settings["remove_binder_monomer"]:
                                    os.remove(mpnn_binder_pdb)

                            # calculate binder averages
                            binder_averages = calculate_averages(binder_statistics)

                            # analyze sequence to make sure there are no cysteins and it contains residues that absorb UV for detection
                            seq_notes = validate_design_sequence(mpnn_sequence['seq'], mpnn_complex_averages.get('Relaxed_Clashes', None), advanced_settings)

                            # measure time to generate design
                            mpnn_end_time = time.time() - mpnn_time
                            elapsed_mpnn_text = f"{'%d hours, %d minutes, %d seconds' % (int(mpnn_end_time // 3600), int((mpnn_end_time % 3600) // 60), int(mpnn_end_time % 60))}"
                            log_progress(f"    ✓ MPNN {mpnn_idx}/{total_mpnn} ({mpnn_design_name}) completed in {int(mpnn_end_time//60)}m {int(mpnn_end_time%60)}s", "MPNN")

                            # Insert statistics about MPNN design into CSV
                            model_numbers = range(1, 6)
                            statistics_labels = ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity',
                                                'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
                                                'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%',
                                                'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Hotspot_RMSD', 'Target_RMSD']

                            # Initialize mpnn_data with the non-statistical data
                            mpnn_data = [mpnn_design_name, advanced_settings["design_algorithm"], length, seed, helicity_value, target_settings["target_hotspot_residues"], mpnn_sequence['seq'], mpnn_interface_residues, mpnn_score, mpnn_seqid]

                            # Add the statistical data for mpnn_complex
                            for label in statistics_labels:
                                mpnn_data.append(mpnn_complex_averages.get(label, None))
                                for model in model_numbers:
                                    mpnn_data.append(mpnn_complex_statistics.get(model, {}).get(label, None))

                            # Add the statistical data for binder
                            for label in ['pLDDT', 'pTM', 'pAE', 'Binder_RMSD']:
                                mpnn_data.append(binder_averages.get(label, None))
                                for model in model_numbers:
                                    mpnn_data.append(binder_statistics.get(model, {}).get(label, None))

                            # Add the remaining non-statistical data
                            mpnn_data.extend([elapsed_mpnn_text, seq_notes, settings_file, filters_file, advanced_file])

                            # insert data into csv
                            insert_data(mpnn_csv, mpnn_data)

                            # find best model number by pLDDT
                            plddt_values = {i: mpnn_data[i] for i in range(11, 15) if mpnn_data[i] is not None}

                            if plddt_values:
                                # Find the key with the highest value
                                highest_plddt_key = int(max(plddt_values, key=plddt_values.get))
                                # Output the number part of the key
                                best_model_number = highest_plddt_key - 10
                                best_model_pdb = os.path.join(design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{best_model_number}.pdb")

                                # run design data against filter thresholds
                                log_progress(f"    Checking filters for {mpnn_design_name}...", "MPNN")
                                filter_conditions = check_filters(mpnn_data, design_labels, filters)
                                if filter_conditions == True:
                                    log_progress(f"    ✓ {mpnn_design_name} passed all filters - ACCEPTED!", "MPNN")
                                    print(f"  ✓ {mpnn_design_name} passed all filters")
                                    accepted_mpnn += 1
                                    accepted_designs += 1

                                    # copy designs to accepted folder
                                    shutil.copy(best_model_pdb, design_paths["Accepted"])

                                    # insert data into final csv
                                    final_data = [''] + mpnn_data
                                    insert_data(final_csv, final_data)

                                    # copy animation from accepted trajectory
                                    if advanced_settings["save_design_animations"]:
                                        accepted_animation = os.path.join(design_paths["Accepted/Animation"], f"{design_name}.html")
                                        if not os.path.exists(accepted_animation):
                                            trajectory_animation = os.path.join(design_paths["Trajectory/Animation"], f"{design_name}.html")
                                            if os.path.exists(trajectory_animation):
                                                shutil.copy(trajectory_animation, accepted_animation)

                                    # copy plots of accepted trajectory
                                    plot_files = os.listdir(design_paths["Trajectory/Plots"])
                                    plots_to_copy = [f for f in plot_files if f.startswith(design_name) and f.endswith('.png')]
                                    for accepted_plot in plots_to_copy:
                                        source_plot = os.path.join(design_paths["Trajectory/Plots"], accepted_plot)
                                        target_plot = os.path.join(design_paths["Accepted/Plots"], accepted_plot)
                                        if not os.path.exists(target_plot):
                                            shutil.copy(source_plot, target_plot)
                                else:
                                    log_progress(f"    ✗ {mpnn_design_name} did not pass filters", "MPNN")
                                    print(f"  ✗ Unmet filter conditions for {mpnn_design_name}")
                                    failure_df = pd.read_csv(failure_csv)
                                    special_prefixes = ('Average_', '1_', '2_', '3_', '4_', '5_')
                                    incremented_columns = set()

                                    for column in filter_conditions:
                                        base_column = column
                                        for prefix in special_prefixes:
                                            if column.startswith(prefix):
                                                base_column = column.split('_', 1)[1]

                                        if base_column not in incremented_columns:
                                            failure_df[base_column] = failure_df[base_column] + 1
                                            incremented_columns.add(base_column)

                                    failure_df.to_csv(failure_csv, index=False)
                                    if os.path.exists(best_model_pdb):
                                        shutil.copy(best_model_pdb, design_paths["Rejected"])

                            # increase MPNN design number
                            mpnn_n += 1

                            # if enough mpnn sequences of the same trajectory pass filters then stop
                            if accepted_mpnn >= advanced_settings["max_mpnn_sequences"]:
                                break

                        if accepted_mpnn >= 1:
                            log_progress(f"  Stage 4: Found {accepted_mpnn} MPNN designs passing filters", "STAGE")
                            print(f"  Found {accepted_mpnn} MPNN designs passing filters")
                        else:
                            log_progress(f"  Stage 4: No accepted MPNN designs found. Evaluated {len(mpnn_sequences)} sequences, all failed filters.", "STAGE")
                            print(f"  No accepted MPNN designs found for this trajectory.")
                            print(f"  Evaluated {len(mpnn_sequences)} MPNN sequences, all failed filters.")

                    else:
                        print('  Duplicate MPNN designs sampled with different trajectory, skipping current trajectory optimisation')

                    # save space by removing unrelaxed design trajectory PDB
                    if advanced_settings["remove_unrelaxed_trajectory"]:
                        os.remove(trajectory_pdb)

                    # measure time it took to generate designs for one trajectory
                    design_time = time.time() - design_start_time
                    design_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(design_time // 3600), int((design_time % 3600) // 60), int(design_time % 60))}"
                    log_progress(f"  ✓ Design and validation of trajectory {design_name} completed in {design_time_text}", "TRAJECTORY")
                    print(f"  Design and validation of trajectory {design_name} took: {design_time_text}")
                    log_progress(f"  Moving to next trajectory...", "TRAJECTORY")
                    print(f"  Moving to next trajectory...")
                
            # analyse the rejection rate of trajectories to see if we need to readjust the design weights
            if trajectory_n >= advanced_settings.get("start_monitoring", 10) and advanced_settings.get("enable_rejection_check", False):
                acceptance = accepted_designs / trajectory_n
                if not acceptance >= advanced_settings.get("acceptance_rate", 0.1):
                    print("⚠ The ratio of successful designs is lower than defined acceptance rate!")
                    print("Consider changing your design settings!")
                    print("Script execution stopping...")
                    break

        # increase trajectory number
        trajectory_n += 1
        
        # Print progress
        num_sampled_trajectories = len(pd.read_csv(trajectory_csv))
        num_accepted_designs = len(pd.read_csv(final_csv))
        log_progress(f"Progress: {num_sampled_trajectories} trajectories sampled, {num_accepted_designs} designs accepted (target: {target_settings['number_of_final_designs']})", "PROGRESS")
        print(f"\nProgress: {num_sampled_trajectories} trajectories sampled, {num_accepted_designs} designs accepted\n")
    
    # Script finished
    elapsed_time = time.time() - script_start_time
    elapsed_text = f"{'%d hours, %d minutes, %d seconds' % (int(elapsed_time // 3600), int((elapsed_time % 3600) // 60), int(elapsed_time % 60))}"
    print(f"\n{'='*60}")
    print(f"Pipeline completed!")
    print(f"Total trajectories: {trajectory_n}")
    print(f"Total time: {elapsed_text}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

