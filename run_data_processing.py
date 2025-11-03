"""
Script to generate processed data from raw MulDiGraph
"""
import sys
import os

# Ensure we're in the right directory
os.chdir(r"d:\Desktop_informations\SGK năm 3\SGK kì 1 năm 3\Visualization -teacher Long\Project final\Data-Prep-Project-main")

# Add current directory to path
sys.path.insert(0, os.getcwd())

from process_data.utils import load_pkl, save_pkl
from process_data.dataset1 import extract_transactions, load_data_muldi, seq_generation, create_phisher_account
import random
from tqdm import tqdm

def main():
    print("=" * 80)
    print("DATA PROCESSING PIPELINE")
    print("=" * 80)
    
    # Check if processed data already exists
    if os.path.exists('process_data/eoa2seq.pkl'):
        print("\n✅ Processed data already exists at: process_data/eoa2seq.pkl")
        print("   If you want to regenerate, please delete the file first.")
        return
    
    print("\n🔄 Step 1: Loading raw MulDiGraph data...")
    graph_file = 'Datas/MulDiGraph.pkl'
    
    if not os.path.exists(graph_file):
        print(f"❌ ERROR: {graph_file} not found!")
        return
    
    graph = load_pkl(graph_file)
    print(f"   ✅ Loaded graph with {graph.number_of_nodes():,} nodes and {graph.number_of_edges():,} edges")
    
    print("\n🔄 Step 2: Extracting transactions from graph...")
    transactions = extract_transactions(graph)
    print(f"   ✅ Extracted {len(transactions):,} transactions")
    
    print("\n🔄 Step 3: Organizing transactions by address...")
    eoa2seq_in, eoa2seq_out = load_data_muldi(transactions)
    print(f"   ✅ Organized transactions for {len(eoa2seq_out):,} addresses")
    
    print("\n🔄 Step 4: Generating transaction sequences...")
    eoa2seq_agg = seq_generation(eoa2seq_in, eoa2seq_out)
    print(f"   ✅ Generated sequences for {len(eoa2seq_agg):,} accounts (filtered: 3-100,000 transactions)")
    
    print("\n🔄 Step 5: Identifying phisher and normal accounts...")
    phisher_accounts, normal_accounts = create_phisher_account(eoa2seq_agg)
    print(f"   ✅ Phisher accounts: {len(phisher_accounts):,}")
    print(f"   ✅ Normal accounts: {len(normal_accounts):,}")
    
    print("\n🔄 Step 6: Saving phisher accounts list...")
    with open("Datas/phisher_accounts.txt", "w") as f:
        for account in phisher_accounts:
            f.write(f"{account}\n")
    print(f"   ✅ Saved to: Datas/phisher_accounts.txt")
    
    print("\n🔄 Step 7: Creating balanced dataset (1:1 ratio)...")
    ratio = 1  # 1:1 ratio of normal to phisher
    random.seed(42)
    selected_normal = random.sample(normal_accounts, min(ratio * len(phisher_accounts), len(normal_accounts)))
    final_accounts = selected_normal + phisher_accounts
    eoa2seq_final = {account: eoa2seq_agg[account] for account in final_accounts}
    print(f"   ✅ Final dataset: {len(eoa2seq_final):,} accounts")
    print(f"      - Normal: {len(selected_normal):,}")
    print(f"      - Phisher: {len(phisher_accounts):,}")
    
    print("\n🔄 Step 8: Saving processed dataset...")
    save_pkl(eoa2seq_final, "process_data/eoa2seq.pkl")
    print(f"   ✅ Saved to: process_data/eoa2seq.pkl")
    
    print("\n🔄 Step 9: Creating address mappings...")
    addresses = set()
    for account, txs in eoa2seq_final.items():
        addresses.add(account)
        for tx in txs:
            addresses.add(tx[0])
    
    address_to_index = {address: idx for idx, address in enumerate(addresses)}
    index_to_address = {idx: address for address, idx in address_to_index.items()}
    
    save_pkl(address_to_index, "process_data/data_Dataset.address_to_index")
    save_pkl(index_to_address, "process_data/data_Dataset.index_to_address")
    print(f"   ✅ Created mappings for {len(addresses):,} unique addresses")
    
    print("\n" + "=" * 80)
    print("✅ DATA PROCESSING COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. process_data/eoa2seq.pkl - Main processed dataset")
    print("  2. Datas/phisher_accounts.txt - List of phisher accounts")
    print("  3. process_data/data_Dataset.address_to_index - Address mappings")
    print("  4. process_data/data_Dataset.index_to_address - Reverse mappings")
    print("\nYou can now run the analysis notebooks!")
    print("=" * 80)

if __name__ == "__main__":
    main()
