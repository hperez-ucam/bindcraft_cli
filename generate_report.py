#!/usr/bin/env python3
"""
Generate PDF report with summary of BindCraft results.
This script creates a pedagogical report with input parameters, top designs, and explanations.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not available. Install with: pip install reportlab")

def get_parameter_explanations():
    """Return dictionary with pedagogical explanations of parameters."""
    return {
        "design_path": {
            "name": "Design Path",
            "explanation": "Output directory where all generated designs and statistics are saved."
        },
        "binder_name": {
            "name": "Binder Name",
            "explanation": "Prefix used to name all generated binder designs. Each design gets a unique identifier based on length and seed."
        },
        "starting_pdb": {
            "name": "Starting PDB",
            "explanation": "Path to the target protein structure file (PDB format). This is the protein that the binder will be designed to interact with."
        },
        "chains": {
            "name": "Target Chains",
            "explanation": "Protein chains to target for binding. Can be a single chain (e.g., 'A') or multiple chains (e.g., 'A,C')."
        },
        "target_hotspot_residues": {
            "name": "Target Hotspot Residues",
            "explanation": "Specific amino acid positions to target for binding. Empty means the algorithm will find the best binding site automatically."
        },
        "lengths": {
            "name": "Binder Length Range",
            "explanation": "Range of amino acid lengths for the designed binder [min, max]. Shorter binders are faster to design but may have lower binding affinity."
        },
        "number_of_final_designs": {
            "name": "Number of Final Designs",
            "explanation": "Target number of successful binder designs to generate. The pipeline will continue until this number is reached."
        },
        "design_protocol": {
            "name": "Design Protocol",
            "explanation": "Algorithm used for binder design. 'Default' uses a 4-stage optimization process (logits → softmax → one-hot → greedy)."
        },
        "prediction_protocol": {
            "name": "Prediction Protocol",
            "explanation": "Method for structure prediction. 'Default' uses standard AlphaFold2 prediction. 'HardTarget' uses initial guess for difficult complexes."
        },
        "interface_protocol": {
            "name": "Interface Protocol",
            "explanation": "Method for interface optimization. 'AlphaFold2' uses AlphaFold2-generated interface. 'MPNN' uses MPNN for interface optimization."
        },
        "template_protocol": {
            "name": "Template Protocol",
            "explanation": "Flexibility of target protein. 'Default' allows limited flexibility. 'Masked' allows greater flexibility at side chain and backbone levels."
        },
        "filter_option": {
            "name": "Filter Option",
            "explanation": "Strictness of quality filters. 'Default' uses recommended filters. 'Relaxed' is more permissive, accepting more designs but potentially with lower quality."
        }
    }

def get_metric_explanations():
    """Return dictionary with explanations of design metrics."""
    return {
        "pLDDT": {
            "name": "pLDDT (Predicted LDDT)",
            "explanation": "Confidence score (0-100) for the predicted structure. Higher values (>70) indicate more reliable predictions. LDDT measures local distance difference test."
        },
        "i_pTM": {
            "name": "Interface pTM",
            "explanation": "Predicted Template Modeling score for the interface region. Measures how well the predicted interface matches expected protein-protein interfaces. Higher is better (>0.5 is good)."
        },
        "i_pLDDT": {
            "name": "Interface pLDDT",
            "explanation": "Confidence score specifically for the interface region where the binder contacts the target protein. Higher values indicate more reliable interface predictions."
        },
        "dG": {
            "name": "Binding Free Energy (ΔG)",
            "explanation": "Calculated binding free energy in kcal/mol. More negative values indicate stronger binding. Typically ranges from -5 to -20 kcal/mol for good binders."
        },
        "dSASA": {
            "name": "Buried Surface Area (ΔSASA)",
            "explanation": "Change in solvent-accessible surface area upon binding (in Å²). Larger values indicate more extensive contact surface, often correlating with stronger binding."
        },
        "dG/dSASA": {
            "name": "Binding Efficiency (ΔG/ΔSASA)",
            "explanation": "Binding free energy per unit of buried surface area. More negative values indicate more efficient binding (stronger interaction per contact area)."
        },
        "ShapeComplementarity": {
            "name": "Shape Complementarity (SC)",
            "explanation": "Measure of how well the binder and target surfaces fit together (0-1 scale). Values >0.6 indicate good shape complementarity, important for specific binding."
        },
        "PackStat": {
            "name": "Packing Statistics",
            "explanation": "Measure of how well atoms are packed at the interface (0-1 scale). Higher values (>0.6) indicate better packing, reducing voids and improving binding."
        },
        "n_InterfaceResidues": {
            "name": "Number of Interface Residues",
            "explanation": "Total number of amino acids involved in the binding interface. More residues can mean stronger binding, but specificity is also important."
        },
        "n_InterfaceHbonds": {
            "name": "Number of Interface Hydrogen Bonds",
            "explanation": "Count of hydrogen bonds formed between binder and target at the interface. More hydrogen bonds generally contribute to stronger and more specific binding."
        },
        "InterfaceHbondsPercentage": {
            "name": "Interface Hydrogen Bond Percentage",
            "explanation": "Percentage of interface residues involved in hydrogen bonding. Higher percentages indicate more polar interactions, often important for specificity."
        },
        "Interface_Hydrophobicity": {
            "name": "Interface Hydrophobicity",
            "explanation": "Measure of hydrophobic character at the interface. Balanced hydrophobicity is important - too much can cause aggregation, too little may reduce binding strength."
        },
        "Relaxed_Clashes": {
            "name": "Relaxed Clashes",
            "explanation": "Number of atomic clashes after structure relaxation. Lower values (ideally 0) indicate better structure quality. Clashes suggest structural problems."
        },
        "Binder_RMSD": {
            "name": "Binder RMSD",
            "explanation": "Root Mean Square Deviation of the binder structure compared to the original trajectory. Lower values indicate the MPNN sequence maintains the designed structure."
        },
        "Target_RMSD": {
            "name": "Target RMSD",
            "explanation": "RMSD of the target protein compared to the input structure. Lower values indicate the target structure is maintained upon binding."
        }
    }

def get_interaction_summary(design_row, metrics_expl):
    """Generate summary of interactions for a design."""
    interactions = []
    
    # Hydrogen bonds
    hbonds = design_row.get('Average_n_InterfaceHbonds', 0)
    hbond_pct = design_row.get('Average_InterfaceHbondsPercentage', 0)
    if pd.notna(hbonds) and hbonds > 0:
        interactions.append(f"• {int(hbonds)} hydrogen bonds ({hbond_pct:.1f}% of interface residues)")
    
    # Hydrophobic interactions
    dg = design_row.get('Average_dG', None)
    if pd.notna(dg) and dg < -5:
        interactions.append(f"• Strong hydrophobic interactions (ΔG = {dg:.2f} kcal/mol)")
    
    # Shape complementarity
    sc = design_row.get('Average_ShapeComplementarity', None)
    if pd.notna(sc) and sc > 0.6:
        interactions.append(f"• Good shape complementarity (SC = {sc:.2f})")
    
    # Packing
    packstat = design_row.get('Average_PackStat', None)
    if pd.notna(packstat) and packstat > 0.6:
        interactions.append(f"• Good atomic packing (PackStat = {packstat:.2f})")
    
    # Surface area
    dsasa = design_row.get('Average_dSASA', None)
    if pd.notna(dsasa) and dsasa > 500:
        interactions.append(f"• Extensive contact surface ({dsasa:.0f} Å² buried)")
    
    if not interactions:
        interactions.append("• Standard protein-protein interactions")
    
    return interactions

def generate_pdf_report(config_path, design_path, output_path=None):
    """Generate PDF report with results summary."""
    if not REPORTLAB_AVAILABLE:
        print("Error: reportlab is required. Install with: pip install reportlab")
        return False
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load final designs
    final_csv = os.path.join(design_path, 'final_design_stats.csv')
    if not os.path.exists(final_csv):
        print(f"Warning: {final_csv} not found. No designs to report.")
        return False
    
    df = pd.read_csv(final_csv)
    if len(df) == 0:
        print("Warning: No accepted designs found.")
        return False
    
    # Sort by pLDDT (or another key metric) and get top 3
    if 'Average_pLDDT' in df.columns:
        df_sorted = df.sort_values('Average_pLDDT', ascending=False)
    else:
        df_sorted = df
    
    top_designs = df_sorted.head(3)
    
    # Set output path
    if output_path is None:
        output_path = os.path.join(design_path, 'BindCraft_Report.pdf')
    
    # Create PDF
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a237e'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#283593'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#3949ab'),
        spaceAfter=8,
        spaceBefore=8
    )
    
    normal_style = styles['Normal']
    normal_style.fontSize = 11
    normal_style.leading = 14
    
    # Title
    story.append(Paragraph("BindCraft Design Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                          ParagraphStyle('Date', parent=normal_style, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.3*inch))
    
    # Section 1: Input Parameters
    story.append(Paragraph("1. Input Parameters and Configuration", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    param_explanations = get_parameter_explanations()
    
    for key, value in config.items():
        if key == "advanced_settings_file" or key == "load_previous_target_settings":
            continue
        
        param_info = param_explanations.get(key, {"name": key, "explanation": "Configuration parameter"})
        param_name = param_info["name"]
        param_expl = param_info["explanation"]
        
        # Format value
        if isinstance(value, list):
            value_str = f"[{', '.join(map(str, value))}]"
        else:
            value_str = str(value)
        
        story.append(Paragraph(f"<b>{param_name}:</b> {value_str}", subheading_style))
        story.append(Paragraph(param_expl, normal_style))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(PageBreak())
    
    # Section 2: Top Designs Summary
    story.append(Paragraph("2. Top Designs Summary", heading_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(f"Total accepted designs: {len(df)}", normal_style))
    story.append(Paragraph(f"Showing top {min(3, len(df))} designs based on pLDDT score.", normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    metrics_expl = get_metric_explanations()
    
    for idx, (_, design) in enumerate(top_designs.iterrows(), 1):
        # Try different possible column names for design name
        design_name = design.get('Design_Name') or design.get('Design') or f'Design_{idx}'
        
        story.append(Paragraph(f"<b>Design {idx}: {design_name}</b>", subheading_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Interactions summary
        story.append(Paragraph("<b>Type of Interactions Established:</b>", normal_style))
        interactions = get_interaction_summary(design, metrics_expl)
        for interaction in interactions:
            story.append(Paragraph(interaction, normal_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Key metrics with explanations
        story.append(Paragraph("<b>Key Parameters and Metrics:</b>", normal_style))
        story.append(Spacer(1, 0.05*inch))
        
        key_metrics = [
            ('Average_pLDDT', 'pLDDT'),
            ('Average_i_pTM', 'i_pTM'),
            ('Average_i_pLDDT', 'i_pLDDT'),
            ('Average_dG', 'dG'),
            ('Average_dSASA', 'dSASA'),
            ('Average_dG/dSASA', 'dG/dSASA'),
            ('Average_ShapeComplementarity', 'ShapeComplementarity'),
            ('Average_PackStat', 'PackStat'),
            ('Average_n_InterfaceResidues', 'n_InterfaceResidues'),
            ('Average_n_InterfaceHbonds', 'n_InterfaceHbonds'),
            ('Average_InterfaceHbondsPercentage', 'InterfaceHbondsPercentage'),
            ('Average_Relaxed_Clashes', 'Relaxed_Clashes')
        ]
        
        for metric_key, metric_name in key_metrics:
            if metric_key in design.index:
                value = design[metric_key]
                if pd.notna(value):
                    metric_info = metrics_expl.get(metric_name, {"name": metric_name, "explanation": "Design metric"})
                    story.append(Paragraph(f"<b>{metric_info['name']}:</b> {value:.3f if isinstance(value, float) else value}", normal_style))
                    story.append(Paragraph(f"  → {metric_info['explanation']}", 
                                         ParagraphStyle('Explanation', parent=normal_style, leftIndent=20, textColor=colors.HexColor('#555555'))))
                    story.append(Spacer(1, 0.05*inch))
        
        if idx < len(top_designs):
            story.append(Spacer(1, 0.2*inch))
    
    story.append(PageBreak())
    
    # Footer with author information
    footer_style = ParagraphStyle(
        'Footer',
        parent=normal_style,
        fontSize=10,
        textColor=colors.HexColor('#666666'),
        alignment=TA_CENTER,
        spaceBefore=0.5*inch
    )
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("─" * 50, footer_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("This BindCraft fork has been created by", footer_style))
    story.append(Paragraph("<b>Horacio Pérez Sánchez</b>", 
                          ParagraphStyle('Author', parent=footer_style, fontSize=12)))
    story.append(Paragraph("Email: hperez@ucam.edu", footer_style))
    story.append(Paragraph("BIO-HPC Research Group: https://bio-hpc.eu", footer_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("─" * 50, footer_style))
    
    # Build PDF
    doc.build(story)
    print(f"✓ PDF report generated: {output_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_report.py <config.json> <design_path> [output.pdf]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    design_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    generate_pdf_report(config_path, design_path, output_path)

