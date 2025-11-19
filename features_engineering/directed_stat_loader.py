import pandas as pd
import numpy as np
from tqdm import tqdm

class directed_loader:

    def __init__(self):
        self.countID = 0
        self.G = {}
        self.co = {}
        self.revco = {}
        self.money = {}

    def nodeID(self, x):
        if x not in self.co:
            self.co[x] = self.countID
            self.revco[self.co[x]] = x
            self.countID += 1
        return self.co[x]

    def read(self, file):
        x = file.values
        for a in range(x.shape[0]):
            i = self.nodeID(x[a, 0])
            j = self.nodeID(x[a, 1])
            self.addEdge((i, j, float(x[a, 2]), float(x[a, 3])))
        self.fixG()

    def storeEmb(self, file, data):
        file1 = open(file, 'w')
        for a in range(data.shape[0]):
            s = str(int(self.revco[a]))
            for b in range(data.shape[1]):
                s += ' ' + str(data[a, b])
            file1.write(s + "\n")
        file1.close()

    def fixG(self):
        for g in range(len(self.G)):
            orderSet = [t for t in self.G[g]]
            orderSet.sort(reverse=True)
            self.G[g] = [(t, np.array([x for x in self.G[g][t]['in']]),
                          np.array([y for y in self.G[g][t]['out']])) for t in orderSet]


    def addEdge(self, s):
        (l1, l2, t, amount) = s
        if l1 not in self.G:
            self.G[l1] = {}
        if l2 not in self.G:
            self.G[l2] = {}
        if t not in self.G[l1]:
            self.G[l1][t] = {'out': set(), 'in': set()}
        if t not in self.G[l2]:
            self.G[l2][t] = {'out': set(), 'in': set()}
        self.G[l1][t]['out'].add(l2)
        self.G[l2][t]['in'].add(l1)

        self.money[l1] = {"incoming_amount": [] , "outgoing_amount": []}
        self.money[l2] = {"incoming_amount": [] , "outgoing_amount": []}
        self.money[l1]["outgoing_amount"].append(amount)
        self.money[l2]["incoming_amount"].append(amount)


    def cal_stat_feats(self):
        """Calculate statistical features for each node in the graph."""
        node_features = {}
        eps = 1e-6  # Small constant to avoid division by zero
        
        for node_id in range(self.countID):
            # Initialize feature dictionary for this node
            features = {
                'node_outdegree': 0,
                'node_indegree': 0,
                'direction_ratio': 0.0,
                'total_val_sent': 0.0,  
                'max_outgoing_amount': 0.0,
                'min_outgoing_amount': float('inf'),
                'max_incoming_amount': 0.0,
                'min_incoming_amount': float('inf'),
                'average_outgoing_amount': 0.0,
                'average_incoming_amount': 0.0,
                'account_balance': 0.0,
                'account_lifetime': 0.0,
                'active_days': 0,
                'mean_hour_sent': 0.0,
                'mean_hour_received': 0.0,
                'std_hour_sent': 0.0,
                'std_hour_received': 0.0,
                'avg_time_between_tx': 0.0,
                'min_time_between_tx': 0.0,
                'max_time_between_tx': 0.0,
                'wd_tx_ratio_sent': 0.0,
                'wd_tx_ratio_received': 0.0
            }

            if node_id not in self.G:
                # If node has no transactions, set default values
                features['min_outgoing_amount'] = 0.0
                features['min_incoming_amount'] = 0.0
                node_features[node_id] = features
                continue
            
            transaction_times = []
            unique_days = set()  # Set to store unique days
            
            # Lists to store temporal information for sent/received transactions
            sent_hours = []
            received_hours = []
            sent_weekdays = []
            received_weekdays = []
            all_tx_times = []
            
            for ti, lii, lio in self.G[node_id]:
                # Convert timestamp to datetime components
                from datetime import datetime
                dt = datetime.fromtimestamp(ti)
                hour = dt.hour
                weekday = dt.weekday()  # 0=Monday, 6=Sunday
                
                # Incoming transactions (received)
                for i in lii:
                    features['node_indegree'] += 1
                    received_hours.append(hour)
                    received_weekdays.append(weekday)
                    all_tx_times.append(ti)
                
                # Outgoing transactions (sent)
                for j in lio:
                    features['node_outdegree'] += 1
                    sent_hours.append(hour)
                    sent_weekdays.append(weekday)
                    all_tx_times.append(ti)

                transaction_times.append(ti)
                
                # Convert timestamp to day
                day = int(ti // 86400)  # 86400 seconds in a day
                unique_days.add(day)

            # Calculate temporal features
            if sent_hours:
                features['mean_hour_sent'] = np.mean(sent_hours)
                features['std_hour_sent'] = np.std(sent_hours) if len(sent_hours) > 1 else 0.0
                
                # Weekend ratio for sent transactions (weekday 5,6 are Sat,Sun)
                weekend_sent = sum(1 for wd in sent_weekdays if wd >= 5)
                features['wd_tx_ratio_sent'] = weekend_sent / len(sent_weekdays)
            
            if received_hours:
                features['mean_hour_received'] = np.mean(received_hours)
                features['std_hour_received'] = np.std(received_hours) if len(received_hours) > 1 else 0.0
                
                # Weekend ratio for received transactions
                weekend_received = sum(1 for wd in received_weekdays if wd >= 5)
                features['wd_tx_ratio_received'] = weekend_received / len(received_weekdays)
            
            # Time between transactions
            if len(all_tx_times) > 1:
                all_tx_times.sort()
                time_diffs = [all_tx_times[i] - all_tx_times[i-1] for i in range(1, len(all_tx_times))]
                
                features['avg_time_between_tx'] = np.mean(time_diffs)
                features['min_time_between_tx'] = min(time_diffs)
                features['max_time_between_tx'] = max(time_diffs)

            transaction_times.sort(reverse=True)
            features["direction_ratio"] = (features['node_indegree'] / (features['node_outdegree'] + eps)) 
            
            # Amount-related features
            if node_id in self.money:
                # Calculate raw values first
                max_out = max(self.money[node_id]["outgoing_amount"]) if self.money[node_id]["outgoing_amount"] else 0.0
                min_out = min(self.money[node_id]["outgoing_amount"]) if self.money[node_id]["outgoing_amount"] else 0.0
                max_in = max(self.money[node_id]["incoming_amount"]) if self.money[node_id]["incoming_amount"] else 0.0
                min_in = min(self.money[node_id]["incoming_amount"]) if self.money[node_id]["incoming_amount"] else 0.0
                avg_out = np.mean(self.money[node_id]["outgoing_amount"]) if self.money[node_id]["outgoing_amount"] else 0.0
                avg_in = np.mean(self.money[node_id]["incoming_amount"]) if self.money[node_id]["incoming_amount"] else 0.0
                balance = sum(self.money[node_id]["incoming_amount"]) - sum(self.money[node_id]["outgoing_amount"])
                
                # Apply bucketization
                features['max_outgoing_amount'] = max_out
                features['min_outgoing_amount'] = min_out
                features["max_incoming_amount"] = max_in
                features["min_incoming_amount"] = min_in
                features["average_outgoing_amount"] = avg_out
                features["average_incoming_amount"] = avg_in
                features["account_balance"] = abs(balance) # Use absolute value for bucketization
            else:
                features['min_outgoing_amount'] = 1  # Bucket value for 0
                features['min_incoming_amount'] = 1  # Bucket value for 0
                features['max_outgoing_amount'] = 1
                features["max_incoming_amount"] = 1
                features["average_outgoing_amount"] = 1
                features["average_incoming_amount"] = 1
                features["account_balance"] = 1
            
            # Time-related features
            features["account_lifetime"] = (transaction_times[0] - transaction_times[-1] if transaction_times else 0.0) / 86400
            features["active_days"] = len(unique_days)  # Number of unique days with transactions
            
            node_features[node_id] = features
        
        return node_features


            
