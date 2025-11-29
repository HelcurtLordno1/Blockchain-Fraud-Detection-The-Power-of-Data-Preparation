"""
Directed Statistical Features Loader

This module calculates 21 statistical features for Ethereum transaction graphs:
- Node degree features (in/out degree, direction ratio)
- Amount-based features (min/max/avg incoming/outgoing amounts, balance)
- Temporal features (account lifetime, active days, transaction timing patterns)
- Weekend transaction ratios

These statistical features capture behavioral patterns that distinguish
normal accounts from phishing/fraudulent accounts.

Author: Fraud Detection Research Team
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Set, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class directed_loader:
    """
    Load and process directed transaction graphs for statistical feature extraction.
    
    This class builds a temporal graph from Ethereum transactions and calculates
    21 statistical features that measure transaction patterns, amounts, and timing.
    
    Attributes:
        countID (int): Counter for assigning unique node IDs
        G (Dict): Graph structure storing temporal edges
        co (Dict): Maps original addresses to integer node IDs
        revco (Dict): Reverse mapping from node IDs to addresses
        money (Dict): Stores incoming/outgoing transaction amounts per node
    
    Example:
        >>> loader = directed_loader()
        >>> loader.read(transaction_df)
        >>> features = loader.cal_stat_feats()
        >>> print(f"Extracted {len(list(features[0].keys()))} features per node")
    """

    def __init__(self) -> None:
        """
        Initialize the directed graph loader.
        
        Sets up empty data structures for node mapping, graph storage,
        and transaction amount tracking.
        """
        self.countID: int = 0
        self.G: Dict[int, Dict] = {}
        self.co: Dict[str, int] = {}
        self.revco: Dict[int, str] = {}
        self.money: Dict[int, Dict[str, list]] = {}

    def nodeID(self, x: str) -> int:
        """
        Get or create a unique integer ID for an Ethereum address.
        
        Args:
            x: Ethereum address string
            
        Returns:
            Unique integer ID for the address
            
        Example:
            >>> loader = directed_loader()
            >>> id1 = loader.nodeID("0xabc123...")
            >>> id2 = loader.nodeID("0xabc123...")  # Same ID returned
            >>> assert id1 == id2
        """
        if x not in self.co:
            self.co[x] = self.countID
            self.revco[self.co[x]] = x
            self.countID += 1
        return self.co[x]

    def read(self, file: pd.DataFrame) -> None:
        """
        Read transaction data and build the directed graph.
        
        Expects DataFrame with columns: [from_address, to_address, timestamp, amount]
        
        Args:
            file: Pandas DataFrame containing transaction records
            
        Example:
            >>> df = pd.DataFrame({
            ...     'from': ['0xabc...', '0xdef...'],
            ...     'to': ['0xdef...', '0xghi...'],
            ...     'timestamp': [1609459200, 1609545600],
            ...     'value': [1.5, 2.3]
            ... })
            >>> loader.read(df)
            >>> logger.info(f"Loaded {len(loader.G)} nodes")
        """
        if file.empty:
            logger.warning("Empty DataFrame provided to read()")
            return
            
        x = file.values
        logger.info(f"Processing {x.shape[0]} transactions with amounts...")
        
        for a in range(x.shape[0]):
            i = self.nodeID(x[a, 0])
            j = self.nodeID(x[a, 1])
            self.addEdge((i, j, float(x[a, 2]), float(x[a, 3])))
        
        self.fixG()
        logger.info(f"Graph built with {self.countID} unique nodes")

    def storeEmb(self, file: str, data: np.ndarray) -> None:
        """
        Store node embeddings/features to file.
        
        Format: Each line contains node_address followed by feature values
        
        Args:
            file: Output file path
            data: NumPy array of shape (n_nodes, n_features)
            
        Example:
            >>> features = loader.cal_stat_feats()
            >>> feature_array = np.array([list(f.values()) for f in features.values()])
            >>> loader.storeEmb("output_statistical_features.txt", feature_array)
        """
        try:
            with open(file, 'w') as file1:
                for a in range(data.shape[0]):
                    s = str(int(self.revco[a]))
                    for b in range(data.shape[1]):
                        s += ' ' + str(data[a, b])
                    file1.write(s + "\n")
            logger.info(f"Stored {data.shape[0]} embeddings with {data.shape[1]} features to {file}")
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")

    def fixG(self) -> None:
        """
        Finalize graph structure by sorting timestamps and converting to arrays.
        
        Converts edge sets to sorted arrays for efficient temporal queries.
        Should be called after all edges are added.
        """
        logger.info("Finalizing graph structure...")
        for g in range(len(self.G)):
            orderSet = [t for t in self.G[g]]
            orderSet.sort(reverse=True)  # Most recent first
            self.G[g] = [(t, np.array([x for x in self.G[g][t]['in']]),
                          np.array([y for y in self.G[g][t]['out']])) for t in orderSet]


    def addEdge(self, s: Tuple[int, int, float, float]) -> None:
        """
        Add a directed edge (transaction) with amount to the graph.
        
        Args:
            s: Tuple of (from_node_id, to_node_id, timestamp, amount)
            
        Example:
            >>> loader.addEdge((1, 2, 1609459200.0, 1.5))  # Node 1 sends 1.5 ETH to Node 2
        """
        (l1, l2, t, amount) = s
        
        # Initialize graph structures
        if l1 not in self.G:
            self.G[l1] = {}
        if l2 not in self.G:
            self.G[l2] = {}
        if t not in self.G[l1]:
            self.G[l1][t] = {'out': set(), 'in': set()}
        if t not in self.G[l2]:
            self.G[l2][t] = {'out': set(), 'in': set()}
        
        # Add edges
        self.G[l1][t]['out'].add(l2)
        self.G[l2][t]['in'].add(l1)

        # Initialize money tracking
        if l1 not in self.money:
            self.money[l1] = {"incoming_amount": [], "outgoing_amount": []}
        if l2 not in self.money:
            self.money[l2] = {"incoming_amount": [], "outgoing_amount": []}
        
        # Track amounts
        self.money[l1]["outgoing_amount"].append(amount)
        self.money[l2]["incoming_amount"].append(amount)


    def cal_stat_feats(self) -> Dict[int, Dict[str, float]]:
        """
        Calculate 21 statistical features for each node in the graph.
        
        Features calculated:
        1. node_outdegree: Number of unique outgoing transactions
        2. node_indegree: Number of unique incoming transactions
        3. direction_ratio: Ratio of incoming to outgoing (fraud indicator)
        4-9. Amount statistics (min/max/avg for incoming/outgoing, balance)
        10-11. Temporal features (account_lifetime, active_days)
        12-15. Hour patterns (mean/std hour sent/received)
        16-18. Time between transactions (avg/min/max)
        19-21. Weekend transaction ratios (sent/received)
        
        Returns:
            Dictionary mapping node_id -> {feature_name: feature_value}
            
        Example:
            >>> loader = directed_loader()
            >>> loader.read(df)
            >>> features = loader.cal_stat_feats()
            >>> node_0_features = features[0]
            >>> print(f"Node 0 indegree: {node_0_features['node_indegree']}")
        """
        logger.info(f"Calculating statistical features for {self.countID} nodes...")
        node_features: Dict[int, Dict[str, float]] = {}
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


            
