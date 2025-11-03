
import random
from tqdm import tqdm
from .utils import *
from math import log
from datetime import datetime
from itertools import combinations
import numpy as np
import argparse

def load_args():
    parser = argparse.ArgumentParser(description="Process dataset arguments")
    parser.add_argument("--ratio", type=int, default= 1, help="Dataset ratio")
    args = parser.parse_args()
    return args

def save_txt(data,txt_file):
    """save data to a text file"""
    with open(txt_file,"w",encoding="utf-8") as file:
        for account in data:
            file.write(f"{account}\n")
    return
                
def cmp_udf_reverse(x1, x2):
    """
    Compare two transactions based on timestamp in descending order
    """
    time1 = int(x1[1])
    time2 = int(x2[1])

    if time1 < time2:
        return 1
    elif time1 > time2:
        return -1
    else:
        return 0
    
def extract_transactions(G):
    """
    Extract transactions from a MultiDiGraph
    """
    transactions = []
    for from_address, to_address, key, tnx_info in tqdm(G.edges(keys=True, data=True),desc=f'accounts_data_generate'):
        amount = tnx_info['amount']
        block_timestamp = int(tnx_info['timestamp'])
        tag = G.nodes[from_address]['isp']
        transaction = {
            'tag': tag,
            'from_address': from_address,
            'to_address': to_address,
            'amount': amount,
            'timestamp': block_timestamp,
        }
        transactions.append(transaction)
    return transactions

def load_data_muldi(transactions):
    """
    Load data for MulDiGraph dataset 
    """
    f_in = {}
    f_out = {}
    error_tran = []
    for tran in transactions: 
        tag = tran['tag']
        from_address = tran['from_address']
        to_address = tran['to_address']
        amount = tran['amount']
        block_timestamp = tran['timestamp']
        if from_address == "" or to_address == "":
            error_tran.append(tran)
            continue
        try:
            f_out[from_address].append([to_address, block_timestamp, amount, "OUT", tag, 1])
        except KeyError:
            f_out[from_address] = [[to_address, block_timestamp, amount, "OUT", tag, 1]]

        try:
            f_in[to_address].append([from_address, block_timestamp, amount, "IN", tag, 1])
        except KeyError:
            f_in[to_address] = [[from_address, block_timestamp, amount, "IN", tag, 1]]

    return f_in, f_out


def seq_generation(eoa2seq_in, eoa2seq_out):
    """
    Generate transaction sequences for each address by merging incoming and outgoing transactions
    """
    eoa_list = list(eoa2seq_out.keys()) # eoa_list must include eoa account only (i.e., have out transaction at least)
    eoa2seq = {}
    for eoa in eoa_list:
        out_seq = eoa2seq_out[eoa]
        try:
            in_seq = eoa2seq_in[eoa]
        except:
            in_seq = []
        seq_agg = sorted(out_seq + in_seq, key= lambda x: int(x[1]))
        cnt_all = 0
        for trans in seq_agg:
            cnt_all += 1
            # if cnt_all >= 5 and cnt_all<=10000:
            if cnt_all > 2 and cnt_all<=100000:
                eoa2seq[eoa] = seq_agg
                break

    return eoa2seq

def create_phisher_account(processed_data):
    """
    Create lists of phisher and normal accounts based on transaction tags
    """
    phisher_accounts = []
    normal_accounts = []
    for address, txs in tqdm(processed_data.items(),desc="Filtering Abnormal vs Normal accounts"):
        i = 0
        for tx in txs:
            if tx[4] == 1:
                phisher_accounts.append(address)
                i += 1
                break
            else: 
                continue
        
        if i == 0: 
            normal_accounts.append(address)
    
    return (phisher_accounts,normal_accounts)

def data_generate():
    """
    Main function to generate processed dataset
    """
    # load and generate data
    args = load_args()
    print("Processing MulDiGraph dataset...")
    graph_file = 'process_data/MulDiGraph.pkl'
    graph = load_pkl(graph_file)
    transactions = extract_transactions(graph)
    eoa2seq_in, eoa2seq_out = load_data_muldi(transactions)
    eoa2seq_agg = seq_generation(eoa2seq_in,eoa2seq_out)

        # Create phisher and normal account lists
    phisher_accounts , normal_accounts = create_phisher_account(eoa2seq_agg)

        # Save phisher accounts to a text file
    print(f"Number of phisher account: {len(phisher_accounts)}")
    with open("phisher_accounts.txt","w") as f:
            for account in phisher_accounts:
                f.write(f"{account}\n")

    # Split into suitable ratio of normal accounts and phisher accounts
    random.seed(42)
    selected_account = random.sample(normal_accounts, args.ratio * len(phisher_accounts))
    final_account = selected_account + phisher_accounts
    eoa2seq_final = {account:eoa2seq_agg[account] for account in final_account} 
    save_pkl(eoa2seq_final, "process_data/eoa2seq.pkl")
    
    # Create address to index and index to address mappings
    addresses = set()
    for account, transactions in eoa2seq_final.items():
        for transaction in transactions:
            from_addr = account
            to_addr = transaction[0]
            if transaction[3] == "IN":
                from_addr, to_addr = to_addr, from_addr
            addresses.add(from_addr)
            addresses.add(to_addr)
    address_to_index = {address: idx for idx, address in enumerate(addresses)}
    index_to_address = {idx: address for address, idx in address_to_index.items()}

    # Save index mappings
    save_pkl(address_to_index, "process_data/data_Dataset.address_to_index")
    save_pkl(index_to_address, "process_data/data_Dataset.index_to_address")

if __name__ == "__main__":
    # Run enhanced data generation
    data_generate()
