import pickle as pkl 
from tqdm import tqdm
import pandas as pd

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data

def save_data(data, file_path):
    with open(file_path, 'wb') as f:
        pkl.dump(data, f)
    return

def save_txt(data, file_path):
    with open(file_path, 'w') as f:
        for tran in data:
            line = ",".join(map(str, tran))
            f.write(str(line) + '\n')
    return

def load_txt(file_path):
    data = pd.read_csv(file_path,names = ["account"])
    return list(data.account.values)

def main():
    data = load_data("./process_data/eoa2seq.pkl")
    print(f"Number of accounts: {len(data)}")
    address_to_index = load_data("./process_data/data_Dataset.address_to_index")
    phisher_account = load_txt("./process_data/phisher_accounts.txt")
    print(f"Number of phisher accounts: {len(phisher_account)}")
    trans = []
    tags = {}
    i = 0
    for eoa, seq in tqdm(data.items(),desc="Processing transactions"):
        if eoa in phisher_account:
            tags[address_to_index[eoa]] = 1
            i += 1
        else:
            tags[address_to_index[eoa]] = 0
        for tx in seq:
            from_addr = eoa
            to_addr = tx[0]
            if tx[3] == "IN":
                from_addr, to_addr = to_addr, from_addr

            trans.append([address_to_index[from_addr],address_to_index[to_addr],tx[1],tx[2]]) # from, to, timestamp, value, other features

    tags = dict(sorted(tags.items(), key=lambda x: x[0]))
    save_data(tags, "./dataset/MulDiGraph/account_tags.pkl")
    save_txt(trans, "./dataset/MulDiGraph/output_transactions.txt")

    return

if __name__ == "__main__":
    main()
    

    

            