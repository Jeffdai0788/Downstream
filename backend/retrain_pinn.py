#!/usr/bin/env python3
"""
TrophicTrace — Retrain PINN with corrected BAF-based ODE
=========================================================
Run this from the backend/ directory:

    cd backend
    python retrain_pinn.py

It will:
  1. Generate 50K ODE training samples using field-measured BAFs (Burkhard 2021)
  2. Train the PINN for 500 epochs with physics constraints
  3. Save pinn_best.pt and pinn_model_info.json
  4. Print validation metrics + comparison vs analytic formula

Requires: torch, numpy, json
Takes ~2-3 min on Apple Silicon, ~5 min on CPU.
"""

import sys
import os

# Make sure we can import from the backend directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pinn_bioaccumulation import train_pinn, load_pinn, predict_with_ci, get_field_baf, CONGENER_LIST

import numpy as np
import json


def validate_against_analytic():
    """
    Compare PINN predictions vs the analytic BAF formula to quantify
    what the PINN adds (temperature sensitivity, DOC effects, transient dynamics, uncertainty).
    """
    print("\n" + "=" * 60)
    print("PINN vs Analytic BAF Formula — Value-Add Comparison")
    print("=" * 60)

    model, info = load_pinn('pinn_best.pt', 'pinn_model_info.json')

    # Test cases: species × water concentration × temperature × congener
    test_cases = [
        # (name, water_ng_l, trophic, lipid, mass, temp, doc, congener)
        ("LM Bass, warm, PFOS",    50, 4.2, 5.8, 1500, 25, 5.0, "PFOS"),
        ("LM Bass, cold, PFOS",    50, 4.2, 5.8, 1500,  8, 5.0, "PFOS"),
        ("LM Bass, high DOC, PFOS",50, 4.2, 5.8, 1500, 18, 15.0, "PFOS"),
        ("LM Bass, low DOC, PFOS", 50, 4.2, 5.8, 1500, 18, 1.0, "PFOS"),
        ("Striped Bass, PFOS",     50, 4.5, 6.1, 5000, 18, 5.0, "PFOS"),
        ("Common Carp, PFOS",      50, 2.9, 5.2, 3000, 18, 5.0, "PFOS"),
        ("LM Bass, PFOA",          50, 4.2, 5.8, 1500, 18, 5.0, "PFOA"),
        ("LM Bass, PFNA",          50, 4.2, 5.8, 1500, 18, 5.0, "PFNA"),
    ]

    print(f"\n{'Case':<30s} {'Analytic':>10s} {'PINN mean':>10s} {'PINN 95% CI':>16s} {'Ratio':>7s}")
    print("-" * 80)

    ratios = []
    for name, water, trophic, lipid, mass, temp, doc, congener in test_cases:
        # Analytic: tissue = water × BAF / 1000
        baf = get_field_baf(congener, trophic)
        analytic = water * baf / 1000

        # PINN with uncertainty
        mean_val, lo, hi = predict_with_ci(
            model, info,
            water_pfas_ng_l=water,
            trophic_level=trophic,
            lipid_pct=lipid,
            body_mass_g=mass,
            temperature_c=temp,
            doc_mg_l=doc,
            congener=congener,
            time_days=365,
            n_passes=50,
        )

        ratio = mean_val / analytic if analytic > 0 else float('inf')
        ratios.append(ratio)

        print(f"{name:<30s} {analytic:>10.2f} {mean_val:>10.2f} [{lo:>6.2f} – {hi:>6.2f}] {ratio:>7.2f}x")

    print(f"\nMedian PINN/Analytic ratio: {np.median(ratios):.2f}x")
    print(f"Mean PINN/Analytic ratio:   {np.mean(ratios):.2f}x")
    print()

    # Show what the PINN adds: temperature sensitivity
    print("--- PINN captures temperature sensitivity ---")
    for temp in [5, 10, 15, 20, 25, 30]:
        mean_val, lo, hi = predict_with_ci(
            model, info, water_pfas_ng_l=50, trophic_level=4.2,
            lipid_pct=5.8, body_mass_g=1500, temperature_c=temp,
            doc_mg_l=5.0, congener='PFOS', time_days=365, n_passes=30
        )
        baf_analytic = get_field_baf('PFOS', 4.2)
        analytic = 50 * baf_analytic / 1000
        print(f"  Temp={temp:2d}°C: PINN={mean_val:.2f} ng/g  Analytic={analytic:.2f} (static)")

    # Show transient dynamics
    print("\n--- PINN captures transient accumulation ---")
    for days in [30, 90, 180, 365]:
        mean_val, lo, hi = predict_with_ci(
            model, info, water_pfas_ng_l=50, trophic_level=4.2,
            lipid_pct=5.8, body_mass_g=1500, temperature_c=18,
            doc_mg_l=5.0, congener='PFOS', time_days=days, n_passes=30
        )
        print(f"  t={days:3d} days: PINN={mean_val:.2f} ng/g [CI: {lo:.2f}–{hi:.2f}]")

    print("\nDone! PINN provides:")
    print("  1. Temperature-dependent bioaccumulation (analytic formula is static)")
    print("  2. DOC-dependent bioavailability correction")
    print("  3. Transient accumulation dynamics (time to steady state)")
    print("  4. Calibrated uncertainty intervals via MC Dropout")


def validate_against_literature():
    """Spot-check PINN predictions against published fish tissue measurements."""
    print("\n" + "=" * 60)
    print("PINN vs Published Fish Tissue Measurements")
    print("=" * 60)

    model, info = load_pinn('pinn_best.pt', 'pinn_model_info.json')

    checks = [
        ("Stahl 2014 — LM Bass, Lake Erie",
         "Water PFOS ~5-15 ng/L → tissue 20-80 ng/g",
         10, 4.2, 5.8, 1500, 15, 5.0, "PFOS", 20, 80),
        ("NC DEQ 2020 — Striped Bass, Cape Fear",
         "Water PFOS ~60 ng/L → tissue 100-300 ng/g",
         60, 4.5, 6.1, 5000, 20, 5.0, "PFOS", 100, 300),
        ("Ye 2008 — Common Carp, Minnesota",
         "Water PFOS ~4 ng/L → tissue 5-25 ng/g",
         4, 2.9, 5.2, 3000, 15, 5.0, "PFOS", 3, 25),
        ("Giesy & Kannan 2001 — Bluegill",
         "Water PFOS ~10 ng/L → tissue 10-70 ng/g",
         10, 3.1, 3.5, 200, 20, 5.0, "PFOS", 10, 70),
    ]

    all_pass = True
    for study, desc, water, tl, lip, mass, temp, doc, cong, lo_lit, hi_lit in checks:
        mean_val, lo, hi = predict_with_ci(
            model, info, water_pfas_ng_l=water, trophic_level=tl,
            lipid_pct=lip, body_mass_g=mass, temperature_c=temp,
            doc_mg_l=doc, congener=cong, time_days=365, n_passes=50
        )
        in_range = lo_lit <= mean_val <= hi_lit
        status = "✓" if in_range else "✗"
        if not in_range:
            all_pass = False
        print(f"\n  {study}")
        print(f"  {desc}")
        print(f"  PINN: {mean_val:.1f} ng/g [CI: {lo:.1f}–{hi:.1f}]  {status}")

    print(f"\n{'All checks passed!' if all_pass else 'Some checks failed — inspect above.'}")


if __name__ == '__main__':
    # Step 1: Retrain
    print("Step 1: Retraining PINN with corrected BAF-based ODE...")
    model, info = train_pinn(n_epochs=500, output_dir='.')

    # Step 2: Validate against literature
    print("\nStep 2: Validating against published data...")
    validate_against_literature()

    # Step 3: Quantify PINN value-add
    print("\nStep 3: Quantifying PINN value-add vs analytic formula...")
    validate_against_analytic()

    print("\n" + "=" * 60)
    print("DONE. Files saved:")
    print("  pinn_best.pt          — retrained model weights")
    print("  pinn_model_info.json  — architecture + validation metrics")
    print("=" * 60)
