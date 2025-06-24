#!/usr/bin/env python3
"""
Script to fix Unknown architecture values in AI Benchmark dataset
Based on GPU names, manufacturers, and web search data about GPU architectures
"""

import pandas as pd
import re
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_architecture_mapping():
    """
    Create a comprehensive mapping of GPU names to their correct architectures
    Based on web search data and GPU architecture information
    """
    
    # NVIDIA Architecture mappings
    nvidia_mappings = {
        # RTX A-series (Ampere architecture for most, some Ada Lovelace)
        'RTX A5000': 'Ampere',
        'RTX A6000': 'Ampere', 
        'RTX A4500': 'Ampere',
        'RTX A4000': 'Ampere',
        'RTX A3000': 'Ampere',
        'RTX A2000': 'Ampere',
        'RTX A4000 Laptop GPU': 'Ampere',
        'RTX A5000 Laptop GPU': 'Ampere',
        'RTX A3000 Laptop GPU': 'Ampere',
        'RTX A2000 Laptop GPU': 'Ampere',
        'RTX A3000 12GB Laptop GPU': 'Ampere',
        'RTX A2000 12GB': 'Ampere',
        
        # Quadro RTX 4000 series (Ada Lovelace)
        'Quadro RTX 4000': 'Ada Lovelace',
        'Quadro RTX 4000 with Max-Q Design': 'Ada Lovelace',
        'Quadro RTX 4000 (Mobile)': 'Ada Lovelace',
        
        # TITAN series
        'TITAN RTX': 'Turing',
        'TITAN Xp': 'Pascal',
        'TITAN Xp COLLECTORS EDITION': 'Pascal', 
        'NVIDIA TITAN Xp': 'Pascal',
        'TITAN V': 'Volta',
        'TITAN V CEO Edition': 'Volta',
        'NVIDIA TITAN X': 'Maxwell',
        'GeForce GTX TITAN X': 'Maxwell',
        'GeForce GTX TITAN Z': 'Kepler',
        'GeForce GTX TITAN Black': 'Kepler',
        'GeForce GTX Titan': 'Kepler',
        
        # Tesla series
        'Tesla T4': 'Turing',
        'Tesla T10': 'Turing',
        'Tesla M40': 'Maxwell',
        'Tesla M40 24GB': 'Maxwell',
        'Tesla M60': 'Maxwell',
        'Tesla M6': 'Maxwell',
        'Tesla P100-PCIE-16GB': 'Pascal',
        'Tesla V100-SXM2-16GB': 'Volta',
        'Tesla K20m': 'Kepler',
        'Tesla K20Xm': 'Kepler',
        
        # T-series (Turing)
        'T1200 Laptop GPU': 'Turing',
        'T1000': 'Turing', 
        'T1000 8GB': 'Turing',
        'T600 Laptop GPU': 'Turing',
        'T600': 'Turing',
        'T550 Laptop GPU': 'Turing',
        
        # P-series mining cards
        'P106-100': 'Pascal',
        
        # GRID series
        'GRID P100-16Q': 'Pascal',
        'GRID M60-8A': 'Maxwell',
        
        # A-series data center
        'A16': 'Ampere',
        'A40-12Q': 'Ampere',
        
        # Other NVIDIA cards
        'Q12U-1': 'Maxwell',
        'GeForce GTX 760 Ti OEM': 'Kepler',
        'GeForce MX550': 'Turing',
        'GeForce RTX 2050': 'Turing',
        'GeForce RTX 3050': 'Ampere',
        'EIZO Quadro MED-XN51LP': 'Pascal',
        'Barco MXRT 7600': 'Pascal',
        
        # GTX 500/600/700 series (Kepler)
        'GeForce GTX 580': 'Fermi',
        'GeForce GTX 570': 'Fermi',
        'GeForce GTX 560': 'Fermi',
        'GeForce GTX 550': 'Fermi',
        'GeForce GTX 480': 'Fermi',
        'GeForce GTX 470': 'Fermi',
        'GeForce GTX 460': 'Fermi',
        'GeForce GTX 680': 'Kepler',
        'GeForce GTX 670': 'Kepler',
        'GeForce GTX 660': 'Kepler',
        'GeForce GTX 650': 'Kepler',
        'GeForce GTX 780': 'Kepler',
        'GeForce GTX 770': 'Kepler',
        'GeForce GTX 760': 'Kepler',
        'GeForce GTX 750': 'Maxwell',
        'GeForce GTX 745': 'Maxwell',
        
        # Mobile GPUs
        'GeForce GTX 880M': 'Kepler',
        'GeForce GTX 870M': 'Kepler',
        'GeForce GTX 860M': 'Kepler',
        'GeForce GTX 850M': 'Maxwell',
        'GeForce GTX 485M': 'Fermi',
        'GeForce GT 645': 'Kepler',
        'GeForce 945M': 'Maxwell',
        'GeForce GTX 590': 'Fermi',
        
        # Tesla C series
        'Tesla C2050': 'Fermi',
        'Tesla C2075': 'Fermi',
        
        # GT series
        'GeForce GT 1030': 'Pascal',
        
        # MX series
        'GeForce MX450': 'Turing',
        'GeForce MX350': 'Pascal',
        'GeForce MX250': 'Pascal',
        'GeForce MX150': 'Pascal',
        'GeForce MX130': 'Pascal',
        
        # GRID RTX series
        'GRID RTX8000-4Q': 'Turing',
        'GRID RTX8000P-2Q': 'Turing',
        'GRID RTX6000-4Q': 'Turing',
        'GRID RTX6000P-4Q': 'Turing',
        'GRID RTX6000P-6Q': 'Turing',
        
        # Legacy 8000 series
        'GeForce 8600 GTS': 'Tesla (Legacy)',
        'GeForce 8800': 'Tesla (Legacy)',
    }
    
    # AMD Architecture mappings  
    amd_mappings = {
        # RDNA 2 (RX 6000 series)
        'Radeon RX 6900 XT': 'RDNA 2',
        'Radeon RX 6800 XT': 'RDNA 2', 
        'Radeon RX 6800': 'RDNA 2',
        'Radeon RX 6700 XT': 'RDNA 2',
        'Radeon RX 6700S': 'RDNA 2',
        'Radeon RX 6700M': 'RDNA 2',
        'Radeon RX 6600 XT': 'RDNA 2',
        'Radeon RX 6600': 'RDNA 2',
        'Radeon RX 6600M': 'RDNA 2',
        'Radeon RX 6500 XT': 'RDNA 2',
        'Radeon RX 6400': 'RDNA 2',
        'Radeon RX 6800M': 'RDNA 2',
        'RadeonT RX 6850M': 'RDNA 2',
        
        # RDNA 2 Pro series
        'Radeon PRO W6800': 'RDNA 2',
        'Radeon PRO W6600': 'RDNA 2', 
        'Radeon PRO W6400': 'RDNA 2',
        
        # GCN Architecture (older cards)
        'Radeon VII': 'GCN (Vega)',
        'Radeon Pro VII': 'GCN (Vega)',
        'Radeon Pro Vega II': 'GCN (Vega)',
        'Radeon Pro Vega 64X': 'GCN (Vega)',
        'Radeon Pro Vega 64': 'GCN (Vega)',
        'Radeon Pro Vega 56': 'GCN (Vega)',
        'Radeon Pro Vega 48': 'GCN (Vega)',
        'Radeon Pro Vega 20': 'GCN (Vega)',
        'Radeon Vega Frontier Edition': 'GCN (Vega)',
        'Radeon R9 Fury X': 'GCN',
        'Radeon R9 Fury': 'GCN',
        'Radeon R9 295X2': 'GCN',
        'Radeon R9 390X': 'GCN',
        'Radeon R9 390': 'GCN',
        'Radeon R9 290X': 'GCN',
        'Radeon R9 290': 'GCN',
        'Radeon R9 285': 'GCN',
        'Radeon Pro Duo': 'GCN',
        'Radeon RX 590': 'GCN',
        'RadeonT RX 5300': 'RDNA',
        'Radeon RX 5300': 'RDNA',
        
        # Professional WX series
        'Radeon Pro WX 8200': 'GCN (Vega)',
        'Radeon Pro WX 9100': 'GCN (Vega)',
        'Radeon Pro WX 7100': 'GCN',
        
        # Pro 5000 series (RDNA)
        'Radeon Pro 5700 XT': 'RDNA',
        'Radeon Pro 5700': 'RDNA',
        'Radeon Pro 5600M': 'RDNA',
        'Radeon Pro 5500 XT': 'RDNA',
        'Radeon Pro 5500M': 'RDNA',
        'Radeon Pro 5300': 'RDNA',
        'Radeon Pro 5300M': 'RDNA',
        
        # Pro 500 series
        'Radeon Pro 580': 'GCN',
        'Radeon Pro 580X': 'GCN',
        'Radeon Pro 570': 'GCN',
        
        # FirePro series
        'FirePro W9100': 'GCN',
        'FirePro W8100': 'GCN',
        'FirePro S9000': 'GCN',
        'FirePro S9050': 'GCN',
        'FirePro S10000': 'GCN',
        'FirePro S7150': 'GCN',
        
        # Instinct series
        'Radeon Instinct MI25 MxGPU': 'GCN (Vega)',
        
        # V520 series
        'Radeon Pro V520 MxGPU': 'RDNA',
        
        # SSG series
        'Radeon Pro SSG': 'GCN (Vega)',
        
        # Ryzen APU graphics (Vega-based)
        'Radeon Ryzen 9 4900HSS': 'GCN (Vega)',
        'Radeon Ryzen 7 4800HS': 'GCN (Vega)', 
        'Radeon Ryzen 9 4900HS': 'GCN (Vega)',
        'Radeon Ryzen 5 4600HS': 'GCN (Vega)',
        'Radeon Ryzen 9 5900HS': 'RDNA 2',
        'Ryzen 7 5800HS with Radeon Graphics': 'RDNA 2',
        'Ryzen 7 4800HS with Radeon Graphics': 'GCN (Vega)',
        'Ryzen 9 5900HS with Radeon Graphics': 'RDNA 2',
        'Radeon Ryzen 7 4800H': 'GCN (Vega)',
        
        # Vega M series
        'Radeon RX Vega M GH': 'GCN (Vega)',
        
        # Other AMD cards 
        'RX 590 GME': 'GCN',
        'Radeon RX590 GME': 'GCN',
        'Radeon RX 580X': 'GCN',
        'Radeon RX 580 2048SP': 'GCN',
        'Radeon Pro W5500': 'RDNA',
        'Radeon Pro W5700': 'RDNA',
        
        # WX series professional cards
        'Radeon Pro WX 7130': 'GCN',
        'Radeon Pro WX 5100': 'GCN',
        'Radeon Pro WX 4100': 'GCN',
        
        # HD series (older GCN and pre-GCN)
        'Radeon HD 7990': 'GCN',
        'Radeon HD 8990': 'GCN',
        'Radeon HD 7970': 'GCN',
        'Radeon HD 7970M': 'GCN',
        'Radeon HD 7870': 'GCN',
        'Radeon HD 7870 XT': 'GCN',
        'Radeon HD 7850': 'GCN',
        'Radeon HD 7800': 'GCN',
        'Radeon HD 7700': 'GCN',
        'Radeon HD 7600': 'GCN',
        'Radeon HD 7500': 'GCN',
        'Radeon HD 8700': 'GCN',
        'Radeon HD 8600': 'GCN',
        'Radeon HD 8500': 'GCN',
        'Radeon HD 8400': 'GCN',
        'Radeon HD 8300': 'GCN',
        'Radeon HD 8200': 'GCN',
        'Radeon HD 6000': 'VLIW5',
        'Radeon HD 5000': 'VLIW5',
        
        # Pro 400/500 series
        'Radeon Pro 465': 'GCN',
        'Radeon Pro 460': 'GCN',
        'Radeon Pro 455': 'GCN',
        'Radeon Pro 450': 'GCN',
        
        # Sky series
        'Radeon Sky 500': 'GCN',
        'Radeon Sky 900': 'GCN',
        
        # Mobile GPUs
        'Radeon HD8970M': 'GCN',
        'Radeon HD8870M': 'GCN',
        'Radeon HD8790M': 'GCN',
        'Radeon HD8670M': 'GCN',
        'Radeon HD7970M': 'GCN',
        'Radeon HD7870M': 'GCN',
        'Radeon HD7770M': 'GCN',
        
        # Embedded GPUs
        'Radeon E8870PCIe': 'GCN',
        'Radeon E9260': 'GCN',
        'Radeon E9550': 'GCN',
        
        # V340 series
        'Radeon Pro V340 MxGPU': 'GCN (Vega)',
        
        # FireStream series
        'FireStream 9370': 'GCN',
        'FireStream 9270': 'GCN',
        
        # Radeon 500 series
        'Radeon 550': 'GCN',
        'Radeon 540': 'GCN',
        'Radeon 530': 'GCN',
        
        # Generic Pro
        'Radeon Pro': 'GCN',
    }
    
    # Intel mappings (less common in this dataset)
    intel_mappings = {
        'Miracast display port driver V3': 'Unknown',  # Keep as unknown - display driver
        '15FF': 'Unknown',  # Generic identifier
    }
    
    return nvidia_mappings, amd_mappings, intel_mappings

def fix_unknown_architectures(input_file, output_file):
    """
    Fix Unknown architecture values in the dataset
    """
    logger.info(f"Loading dataset from {input_file}")
    df = pd.read_csv(input_file)
    
    # Get architecture mappings
    nvidia_mappings, amd_mappings, intel_mappings = create_architecture_mapping()
    
    # Count original unknown values
    unknown_count_before = (df['Architecture'] == 'Unknown').sum()
    logger.info(f"Found {unknown_count_before} Unknown architecture values")
    
    # Track changes
    changes_made = []
    
    # Process each row with Unknown architecture
    for idx, row in df.iterrows():
        if row['Architecture'] == 'Unknown':
            gpu_name = row['gpuName']
            manufacturer = row['Manufacturer']
            
            new_architecture = None
            
            # Check manufacturer-specific mappings
            if manufacturer == 'NVIDIA':
                # Check direct mapping first
                if gpu_name in nvidia_mappings:
                    new_architecture = nvidia_mappings[gpu_name]
                else:
                    # Pattern matching for NVIDIA cards
                    if 'RTX A' in gpu_name and 'Laptop' in gpu_name:
                        new_architecture = 'Ampere'
                    elif 'RTX A' in gpu_name:
                        new_architecture = 'Ampere'
                    elif 'Quadro RTX 4000' in gpu_name:
                        new_architecture = 'Ada Lovelace'
                    elif 'Tesla T' in gpu_name:
                        new_architecture = 'Turing'
                    elif 'Tesla M' in gpu_name:
                        new_architecture = 'Maxwell'
                    elif 'Tesla P' in gpu_name:
                        new_architecture = 'Pascal'
                    elif 'Tesla V' in gpu_name:
                        new_architecture = 'Volta'
                    elif 'Tesla K' in gpu_name:
                        new_architecture = 'Kepler'
                    elif 'Tesla C' in gpu_name:
                        new_architecture = 'Fermi'
                    elif 'GRID RTX' in gpu_name:
                        new_architecture = 'Turing'
                    elif 'GTX 590' in gpu_name:
                        new_architecture = 'Fermi'
                    elif 'GTX 580' in gpu_name or 'GTX 570' in gpu_name or 'GTX 560' in gpu_name or 'GTX 550' in gpu_name:
                        new_architecture = 'Fermi'
                    elif 'GTX 480' in gpu_name or 'GTX 470' in gpu_name or 'GTX 460' in gpu_name:
                        new_architecture = 'Fermi'
                    elif 'GTX 780' in gpu_name or 'GTX 770' in gpu_name or 'GTX 760' in gpu_name:
                        new_architecture = 'Kepler'
                    elif 'GTX 680' in gpu_name or 'GTX 670' in gpu_name or 'GTX 660' in gpu_name or 'GTX 650' in gpu_name:
                        new_architecture = 'Kepler'
                    elif 'GTX 750' in gpu_name or 'GTX 745' in gpu_name:
                        new_architecture = 'Maxwell'
                    elif 'GTX 880M' in gpu_name or 'GTX 870M' in gpu_name or 'GTX 860M' in gpu_name:
                        new_architecture = 'Kepler'
                    elif 'GTX 850M' in gpu_name:
                        new_architecture = 'Maxwell'
                    elif 'GTX 485M' in gpu_name:
                        new_architecture = 'Fermi'
                    elif 'GT 1030' in gpu_name:
                        new_architecture = 'Pascal'
                    elif 'GT 645' in gpu_name:
                        new_architecture = 'Kepler'
                    elif '945M' in gpu_name:
                        new_architecture = 'Maxwell'
                    elif 'MX' in gpu_name and ('450' in gpu_name or '550' in gpu_name):
                        new_architecture = 'Turing'
                    elif 'MX' in gpu_name:
                        new_architecture = 'Pascal'
                    elif 'GeForce 8' in gpu_name:
                        new_architecture = 'Tesla (Legacy)'
                    elif 'TITAN' in gpu_name and ('Xp' in gpu_name or 'Pascal' in str(row.get('testDate', ''))):
                        new_architecture = 'Pascal'
                    elif 'TITAN V' in gpu_name:
                        new_architecture = 'Volta'
                    elif 'TITAN RTX' in gpu_name:
                        new_architecture = 'Turing'
                    elif 'TITAN X' in gpu_name or 'TITAN Z' in gpu_name or 'TITAN Black' in gpu_name:
                        new_architecture = 'Maxwell' if 'TITAN X' in gpu_name else 'Kepler'
                    elif 'T' in gpu_name and ('600' in gpu_name or '1000' in gpu_name or '1200' in gpu_name):
                        new_architecture = 'Turing'
                        
            elif manufacturer == 'AMD':
                # Check direct mapping first
                if gpu_name in amd_mappings:
                    new_architecture = amd_mappings[gpu_name]
                else:
                    # Pattern matching for AMD cards
                    if 'RX 6' in gpu_name:
                        new_architecture = 'RDNA 2'
                    elif 'RX 5' in gpu_name:
                        new_architecture = 'RDNA'
                    elif 'Vega' in gpu_name:
                        new_architecture = 'GCN (Vega)'
                    elif 'R9' in gpu_name or 'RX 590' in gpu_name or 'RX 580' in gpu_name or 'RX 480' in gpu_name or 'RX 470' in gpu_name:
                        new_architecture = 'GCN'
                    elif 'HD 7' in gpu_name or 'HD 8' in gpu_name or 'HD7' in gpu_name or 'HD8' in gpu_name:
                        new_architecture = 'GCN'
                    elif 'HD 6' in gpu_name or 'HD 5' in gpu_name or 'HD6' in gpu_name or 'HD5' in gpu_name:
                        new_architecture = 'VLIW5'
                    elif 'FirePro' in gpu_name:
                        new_architecture = 'GCN'
                    elif 'FireStream' in gpu_name:
                        new_architecture = 'GCN'
                    elif 'Pro V340' in gpu_name:
                        new_architecture = 'GCN (Vega)'
                    elif 'Pro W' in gpu_name and '6' in gpu_name:
                        new_architecture = 'RDNA 2'
                    elif 'Pro W' in gpu_name:
                        new_architecture = 'GCN'
                    elif 'Pro 5' in gpu_name:
                        new_architecture = 'RDNA'
                    elif 'Pro 4' in gpu_name:
                        new_architecture = 'GCN'
                    elif 'Pro' in gpu_name and ('580' in gpu_name or '570' in gpu_name or '560' in gpu_name or '550' in gpu_name):
                        new_architecture = 'GCN'
                    elif 'Radeon 5' in gpu_name:
                        new_architecture = 'GCN'
                    elif 'Sky' in gpu_name:
                        new_architecture = 'GCN'
                    elif 'Radeon E' in gpu_name:
                        new_architecture = 'GCN'
                    elif 'Ryzen' in gpu_name:
                        # Ryzen APUs - check generation
                        if '5900' in gpu_name or '5800' in gpu_name:
                            new_architecture = 'RDNA 2'
                        else:
                            new_architecture = 'GCN (Vega)'
                    elif 'Instinct' in gpu_name:
                        new_architecture = 'GCN (Vega)'
                        
            elif manufacturer == 'Intel':
                if gpu_name in intel_mappings:
                    new_architecture = intel_mappings[gpu_name]
                # Most Intel entries in this dataset are display drivers, keep as Unknown
            
            # Apply the change if we found a mapping
            if new_architecture and new_architecture != 'Unknown':
                df.at[idx, 'Architecture'] = new_architecture
                changes_made.append({
                    'gpu_name': gpu_name,
                    'manufacturer': manufacturer,
                    'old_architecture': 'Unknown',
                    'new_architecture': new_architecture
                })
                logger.info(f"Updated {gpu_name} ({manufacturer}): Unknown → {new_architecture}")
    
    # Count remaining unknown values
    unknown_count_after = (df['Architecture'] == 'Unknown').sum()
    fixed_count = unknown_count_before - unknown_count_after
    
    logger.info(f"Fixed {fixed_count} architecture values")
    logger.info(f"Remaining Unknown values: {unknown_count_after}")
    
    # Save the updated dataset
    logger.info(f"Saving updated dataset to {output_file}")
    df.to_csv(output_file, index=False)
    
    # Create a summary report
    if changes_made:
        changes_df = pd.DataFrame(changes_made)
        summary_file = output_file.parent / f"architecture_fixes_summary.csv"
        changes_df.to_csv(summary_file, index=False)
        logger.info(f"Summary of changes saved to {summary_file}")
        
        # Print summary by manufacturer
        print("\n=== ARCHITECTURE FIX SUMMARY ===")
        for manufacturer in changes_df['manufacturer'].unique():
            mfg_changes = changes_df[changes_df['manufacturer'] == manufacturer]
            print(f"\n{manufacturer}: {len(mfg_changes)} fixes")
            for arch in mfg_changes['new_architecture'].unique():
                count = len(mfg_changes[mfg_changes['new_architecture'] == arch])
                print(f"  → {arch}: {count} cards")
    
    return df, changes_made

def main():
    """Main function"""
    # Define file paths
    input_file = Path("data/final/Ai-Benchmark-Final-enhanced.csv")
    output_file = Path("data/final/Ai-Benchmark-Final-enhanced-fixed.csv")
    
    # Ensure input file exists
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Fix the architectures
    try:
        df, changes = fix_unknown_architectures(input_file, output_file)
        logger.info("Architecture fixes completed successfully!")
        
        # Final statistics
        total_records = len(df)
        unknown_remaining = (df['Architecture'] == 'Unknown').sum()
        known_architectures = total_records - unknown_remaining
        
        print(f"\n=== FINAL STATISTICS ===")
        print(f"Total records: {total_records:,}")
        print(f"Known architectures: {known_architectures:,} ({known_architectures/total_records*100:.1f}%)")
        print(f"Unknown architectures: {unknown_remaining:,} ({unknown_remaining/total_records*100:.1f}%)")
        print(f"Changes made: {len(changes)}")
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise

if __name__ == "__main__":
    main() 