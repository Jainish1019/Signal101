#!/usr/bin/env python3
"""Step 8: Run Antigravity agents to generate proof pack."""
import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import agent modules directly
from src.agents.agent1_integrity import run_integrity_audit
from src.agents.agent2_dataset_qa import run_dataset_qa
from src.agents.agent3_evaluation import run_evaluation_reporter
from src.agents.agent4_failure import run_failure_miner

def run():
    print("=" * 60)
    print("STEP 8: GOOGLE ANTIGRAVITY WORKFLOW (PROOF PACK)")
    print("=" * 60)
    print("Dispatching autonomous agents to evaluate pipeline...\n")
    
    proof_dir = PROJECT_ROOT / "proof_pack"
    proof_dir.mkdir(exist_ok=True)
    
    run_integrity_audit()
    run_dataset_qa()
    run_evaluation_reporter()
    run_failure_miner()
    
    print("\n[+] Proof pack generation complete. Artifacts saved to ./proof_pack/")

if __name__ == "__main__":
    run()
