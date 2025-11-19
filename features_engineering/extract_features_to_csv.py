import argparse
import pandas as pd
import numpy as np
import directed_freq_loader
import directed_stat_loader
import networkx as nx
import scipy.sparse as sp
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Extract all features and export to CSV')
    parser.add_argument("--dataset", type=str, default="MulDiGraph", 
                        help="Dataset to use (default: MulDiGraph)")
    parser.add_argument("--input", type=str, default=None,
                        help="Input graph path (default: ./dataset/<dataset>/output_transactions.txt)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: ./dataset/<dataset>/features.csv)")
    parser.add_argument("--alpha", type=float, default=1.0, 
                        help="Temporal decay factor for centrality calculation")
    
    args = parser.parse_args()
    
    if args.input is None:
        args.input = f"./dataset/{args.dataset}/output_transactions.txt"
    if args.output is None:
        args.output = f"./dataset/{args.dataset}/features.csv"
        
    return args

def construct_directed_adjacency_matrix(G, nv, alpha):
    """Construct directed adjacency matrix preserving edge directions."""
    row, col, data = [], [], []
    
    for v in range(nv):
        for (t, lii, lio) in G[v]:
            weight = np.exp(-t / alpha)
            
            lii_list = lii.tolist() if lii is not None and lii.size > 0 else []
            for u in lii_list:
                row.append(u)
                col.append(v)
                data.append(weight)
            
            lio_list = lio.tolist() if lio is not None and lio.size > 0 else []
            for u in lio_list:
                row.append(v)
                col.append(u)
                data.append(weight)
    
    A_directed = sp.csr_matrix((data, (row, col)), shape=(nv, nv))
    A_directed = (A_directed > 0).astype(np.float32)
    
    return A_directed

def compute_centrality_features(G, nv, alpha):
    """Compute centrality features efficiently."""
    A_directed = construct_directed_adjacency_matrix(G, nv, alpha)
    A_nx_directed = nx.from_scipy_sparse_array(A_directed, parallel_edges=False, 
                                               edge_attribute="weight", create_using=nx.DiGraph)
    
    features = {}
    
    print(f"Computing centrality features for {nv} nodes...")
    
    print("Computing Katz centrality...")
    try:
        features['katz'] = nx.katz_centrality(A_nx_directed, alpha=0.01, beta=1.0, max_iter=1000, tol=1e-6)
    except:
        features['katz'] = {i: 0.0 for i in range(nv)}

    print("Computing Degree centrality...")
    features['degree'] = nx.degree_centrality(A_nx_directed)

    print("Computing Closeness centrality...")
    features['closeness'] = nx.closeness_centrality(A_nx_directed)

    print("Computing Clustering coefficient...")
    features['clustering'] = nx.clustering(A_nx_directed)

    print("Computing Eigenvector centrality...")
    try:
        features['eigenvector'] = nx.eigenvector_centrality(A_nx_directed, max_iter=1000, tol=1e-4)
    except:
        features['eigenvector'] = nx.pagerank(A_nx_directed, max_iter=1000, tol=1e-4)

    print("Computing In-degree centrality...")
    features['indegree'] = nx.in_degree_centrality(A_nx_directed)
    
    print("Computing Out-degree centrality...")
    features['outdegree'] = nx.out_degree_centrality(A_nx_directed)
    
    print("Centrality computation completed.")
    return features

def main():
    args = parse_args()
    
    # Read data
    print("Reading input data...")
    data = pd.read_csv(args.input)
    
    # Initialize loaders
    print("Initializing frequency loader...")
    freq_loader = directed_freq_loader.directed_loader()
    freq_loader.read(data)
    nv = len(freq_loader.G)
    
    print("Initializing statistical loader...")
    stat_loader = directed_stat_loader.directed_loader()
    stat_loader.read(data)
    
    # Calculate features
    print("Calculating frequency features...")
    freq_features = freq_loader.cal_stat_feats()
    print(f"Frequency features calculated for {len(freq_features)} nodes")
    
    print("Calculating statistical features...")
    stat_features = stat_loader.cal_stat_feats()
    print(f"Statistical features calculated for {len(stat_features)} nodes")
    
    print("Computing graph centrality features...")
    graph_features = compute_centrality_features(freq_loader.G, nv, args.alpha)
    
    # Prepare data for CSV
    print("Preparing data for CSV export...")
    rows = []
    
    for v in tqdm(range(nv), desc="Processing nodes"):
        row = {'node_id': int(freq_loader.revco[v])}
        
        # Add frequency features (6 dimensions)
        if v in freq_features:
            row['long_term_transfer_freq'] = freq_features[v]['Long-term transfer frequency']
            row['short_term_transfer_freq'] = freq_features[v]['Short-term transfer frequency']
            row['long_term_incoming_freq'] = freq_features[v]['Long-term incoming transfer frequency']
            row['short_term_incoming_freq'] = freq_features[v]['Short-term incoming transfer frequency']
            row['long_term_outgoing_freq'] = freq_features[v]['Long-term outgoing transfer frequency']
            row['short_term_outgoing_freq'] = freq_features[v]['Short-term outgoing transfer frequency']
        else:
            row['long_term_transfer_freq'] = 0.0
            row['short_term_transfer_freq'] = 0.0
            row['long_term_incoming_freq'] = 0.0
            row['short_term_incoming_freq'] = 0.0
            row['long_term_outgoing_freq'] = 0.0
            row['short_term_outgoing_freq'] = 0.0
        
        # Add statistical features (selected 19 dimensions based on your code)
        if v in stat_features:
            # Only the uncommented features from your selection
            row['node_indegree'] = stat_features[v]['node_indegree']
            row['direction_ratio'] = stat_features[v]['direction_ratio']
            row['max_outgoing_amount'] = stat_features[v]['max_outgoing_amount']
            row['min_outgoing_amount'] = stat_features[v]['min_outgoing_amount']
            row['average_outgoing_amount'] = stat_features[v]['average_outgoing_amount']
            row['account_balance'] = stat_features[v]['account_balance']
            row['mean_hour_received'] = stat_features[v]['mean_hour_received']
            row['std_hour_received'] = stat_features[v]['std_hour_received']
            row['min_time_between_tx'] = stat_features[v]['min_time_between_tx']
            row['wd_tx_ratio_received'] = stat_features[v]['wd_tx_ratio_received']
            
            # Additional statistical features that were commented out (for completeness)
            row['node_outdegree'] = stat_features[v]['node_outdegree']
            row['max_incoming_amount'] = stat_features[v]['max_incoming_amount']
            row['min_incoming_amount'] = stat_features[v]['min_incoming_amount']
            row['average_incoming_amount'] = stat_features[v]['average_incoming_amount']
            row['account_lifetime'] = stat_features[v]['account_lifetime']
            row['active_days'] = stat_features[v]['active_days']
            row['mean_hour_sent'] = stat_features[v]['mean_hour_sent']
            row['std_hour_sent'] = stat_features[v]['std_hour_sent']
            row['avg_time_between_tx'] = stat_features[v]['avg_time_between_tx']
            row['max_time_between_tx'] = stat_features[v]['max_time_between_tx']
            row['wd_tx_ratio_sent'] = stat_features[v]['wd_tx_ratio_sent']
        else:
            # Default values if node not found in stat_features
            stat_cols = ['node_indegree', 'direction_ratio', 'max_outgoing_amount', 'min_outgoing_amount',
                        'average_outgoing_amount', 'account_balance', 'mean_hour_received', 'std_hour_received',
                        'min_time_between_tx', 'wd_tx_ratio_received', 'node_outdegree', 'max_incoming_amount',
                        'min_incoming_amount', 'average_incoming_amount', 'account_lifetime', 'active_days',
                        'mean_hour_sent', 'std_hour_sent', 'avg_time_between_tx', 'max_time_between_tx', 'wd_tx_ratio_sent']
            for col in stat_cols:
                row[col] = 0.0
        
        # Add centrality features (7 dimensions)
        row['katz_centrality'] = graph_features['katz'].get(v, 0.0)
        row['degree_centrality'] = graph_features['degree'].get(v, 0.0)
        row['closeness_centrality'] = graph_features['closeness'].get(v, 0.0)
        row['clustering_coefficient'] = graph_features['clustering'].get(v, 0.0)
        row['eigenvector_centrality'] = graph_features['eigenvector'].get(v, 0.0)
        row['indegree_centrality'] = graph_features['indegree'].get(v, 0.0)
        row['outdegree_centrality'] = graph_features['outdegree'].get(v, 0.0)
        
        rows.append(row)
    
    # Create DataFrame and save to CSV
    print("Creating DataFrame and saving to CSV...")
    df = pd.DataFrame(rows)
    
    # Reorder columns for better readability
    freq_cols = ['long_term_transfer_freq', 'short_term_transfer_freq', 'long_term_incoming_freq', 
                'short_term_incoming_freq', 'long_term_outgoing_freq', 'short_term_outgoing_freq']
    
    selected_stat_cols = ['node_indegree', 'direction_ratio', 'max_outgoing_amount', 'min_outgoing_amount',
                         'average_outgoing_amount', 'account_balance', 'mean_hour_received', 'std_hour_received',
                         'min_time_between_tx', 'wd_tx_ratio_received']
    
    other_stat_cols = ['node_outdegree', 'max_incoming_amount', 'min_incoming_amount', 'average_incoming_amount',
                      'account_lifetime', 'active_days', 'mean_hour_sent', 'std_hour_sent', 'avg_time_between_tx', 'max_time_between_tx', 'wd_tx_ratio_sent']
    
    centrality_cols = ['katz_centrality', 'degree_centrality', 'closeness_centrality', 'clustering_coefficient',
                      'eigenvector_centrality', 'indegree_centrality', 'outdegree_centrality']
    
    column_order = ['node_id'] + freq_cols + selected_stat_cols + other_stat_cols + centrality_cols
    df = df[column_order]
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    
    print(f"Features exported to {args.output}")
    print(f"Dataset shape: {df.shape}")
    print("\nColumn summary:")
    print(f"- Node ID: 1 column")
    print(f"- Frequency features: {len(freq_cols)} columns")
    print(f"- Selected statistical features: {len(selected_stat_cols)} columns")
    print(f"- Other statistical features: {len(other_stat_cols)} columns")
    print(f"- Centrality features: {len(centrality_cols)} columns")
    print(f"- Total features: {len(df.columns)-1} columns")
    
    # Show first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Show feature statistics
    print("\nFeature statistics:")
    print(df.describe())

if __name__ == "__main__":
    main()