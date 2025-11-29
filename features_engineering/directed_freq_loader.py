import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any
from tqdm import tqdm
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class directed_loader:

    def __init__(self) -> None:
        """
        Initialize the directed graph loader.
        
        Sets up empty data structures for node mapping and graph storage.
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
        
        Expects DataFrame with columns: [from_address, to_address, timestamp]
        
        Args:
            file: Pandas DataFrame containing transaction records
            
        Example:
            >>> df = pd.DataFrame({
            ...     'from': ['0xabc...', '0xdef...'],
            ...     'to': ['0xdef...', '0xghi...'],
            ...     'timestamp': [1609459200, 1609545600]
            ... })
            >>> loader.read(df)
            >>> logger.info(f"Loaded {len(loader.G)} nodes")
        """
        if file.empty:
            logger.warning("Empty DataFrame provided to read()")
            return
            
        x = file.values
        logger.info(f"Processing {x.shape[0]} transactions...")
        
        for a in range(x.shape[0]):
            i = self.nodeID(x[a, 0])
            j = self.nodeID(x[a, 1])
            ts = float(x[a, 2])
            amount = float(x[a, 3]) if x.shape[1] > 3 else 0.0
            self.addEdge((i, j, ts, amount))
        
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
            >>> loader.storeEmb("output_features.txt", feature_array)
        """
        try:
            with open(file, 'w') as file1:
                for a in range(data.shape[0]):
                    s = str(int(self.revco[a]))
                    for b in range(data.shape[1]):
                        s += ' ' + str(data[a, b])
                    file1.write(s + "\n")
            logger.info(f"Stored {data.shape[0]} embeddings to {file}")
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
        Add a directed edge (transaction) to the graph.
        
        Args:
            s: Tuple of (from_node_id, to_node_id, timestamp, amount)
            
        Example:
            >>> loader.addEdge((1, 2, 1609459200.0, 1.5))  # Node 1 -> Node 2 at timestamp with amount
        """
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

        # Initialize money buckets if not present
        if l1 not in self.money:
            self.money[l1] = {"incoming_amount": [], "outgoing_amount": []}
        if l2 not in self.money:
            self.money[l2] = {"incoming_amount": [], "outgoing_amount": []}
        
        self.money[l1]["outgoing_amount"].append(amount)
        self.money[l2]["incoming_amount"].append(amount)


    def cal_stat_feats(self) -> Dict[int, Dict[str, float]]:
        """Calculate frequency features for each node in the graph."""
        node_features: Dict[int, Dict[str, float]] = {}
        eps = 1e-6  # Small constant to avoid division by zero
        
        times = set()
        for node_id in range(self.countID):
            if node_id in self.G:
                for ti, _, _ in self.G[node_id]:
                    times.add(ti)

        if not times:
            # Return empty features if no transactions
            return {nid: {
                'Long-term transfer frequency': 0.0,
                'Short-term transfer frequency': 0.0,
                'Long-term incoming transfer frequency': 0.0,
                'Short-term incoming transfer frequency': 0.0,
                'Long-term outgoing transfer frequency': 0.0,
                'Short-term outgoing transfer frequency': 0.0
            } for nid in range(self.countID)}

        times = sorted(times, reverse=True) 
        latest = times[0]
        short_term_windows = latest - timedelta(days=7).total_seconds()
        long_term_windows = latest - timedelta(days=30).total_seconds()       

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

            if node_id not in self.G:
                node_features[node_id] = features
                continue

            for ti, innodes, outnodes in self.G[node_id]:
                if ti >= short_term_windows:
                    short_term_count["in"] += int(len(innodes))
                    short_term_count["out"] += int(len(outnodes))
                if ti >= long_term_windows:
                    long_term_count["in"] += int(len(innodes))
                    long_term_count["out"] += int(len(outnodes))

            features['Long-term transfer frequency'] = (long_term_count["in"] + long_term_count["out"]) / (30 + eps)
            features['Short-term transfer frequency'] = (short_term_count["in"] + short_term_count["out"]) / (7 + eps)
            features['Long-term incoming transfer frequency'] = long_term_count["in"] / (30 + eps)
            features['Short-term incoming transfer frequency'] = short_term_count["in"] / (7 + eps)
            features['Long-term outgoing transfer frequency'] = long_term_count["out"] / (30 + eps)
            features['Short-term outgoing transfer frequency'] = short_term_count["out"] / (7 + eps)
            node_features[node_id] = features
            
        return node_features


            
