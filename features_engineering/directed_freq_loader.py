import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

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
        
        times = set()
        for node_id in range(self.countID):
            for ti, _ ,_ in self.G[node_id]:
                times.add(ti)

        times = sorted(times,reverse=True) 

        short_term_windows = times[0] - timedelta(days = 7).total_seconds()
        long_term_windows = times[0] - timedelta(days = 30).total_seconds()       

        for node_id in range(self.countID):
            # Initialize feature dictionary for this node
            features = {
               'Long-term transfer frequency': 0.0,
               'Short-term transfer frequency': 0.0,
               'Long-term incoming transfer frequency': 0.0,
               'Short-term incoming transfer frequency': 0.0,
               'Long-term outgoing transfer frequency': 0.0,
               'Short-term outgoing transfer frequency': 0.0
            }

            long_term_count = {"in": 0, "out": 0}
            short_term_count = {"in": 0, "out": 0}

            for ti, innodes, outnodes in self.G[node_id]:
                if ti >= short_term_windows:
                    short_term_count["in"] += len(innodes)
                    short_term_count["out"] += len(outnodes)
                if ti >= long_term_windows:
                    long_term_count["in"] += len(innodes)
                    long_term_count["out"] += len(outnodes)

            features['Long-term transfer frequency'] = (long_term_count["in"] + long_term_count["out"]) / (30 + eps)
            features['Short-term transfer frequency'] = (short_term_count["in"] + short_term_count["out"]) / (7 + eps)
            features['Long-term incoming transfer frequency'] = long_term_count["in"] / (30 + eps)
            features['Short-term incoming transfer frequency'] = short_term_count["in"] / (7 + eps)
            features['Long-term outgoing transfer frequency'] = long_term_count["out"] / (30 + eps)
            features['Short-term outgoing transfer frequency'] = short_term_count["out"] / (7 + eps)
            node_features[node_id] = features
            
        return node_features


            
