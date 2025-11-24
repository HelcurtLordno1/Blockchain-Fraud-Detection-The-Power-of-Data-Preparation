import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import directed_freq_loader
import directed_stat_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse.linalg import cg
import psutil
import networkx as nx
import sys
import pickle
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description='Directed Temporal SIRGN with Laplacian Optimization')
    parser.add_argument("--dataset", type=str, default="MulDiGraph", choices=["B4E", "MulDiGraph", "TXNT"],
                        help="Dataset to use (default: MulDiGraph)")
    parser.add_argument("--input", type=str, default=None,
                        help="Input graph path (default: ../data/dataset/MulDiGraph/output_transactions.txt)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output embedding path (default: ../data/dataset/Data_after_FE/graph_emb.txt)")
    parser.add_argument("--depth", type=int, default=10, help="Number of iterations")
    parser.add_argument("--alpha", type=float, default=1.0, help="Temporal decay factor")
    parser.add_argument("--clusters", type=int, default=10, help="Number of clusters")
    parser.add_argument("--stop", default=True, action="store_true", help="Stop at convergence")
    parser.add_argument("--beta", type=float, default=10.0, help="Inverse temperature for K-means")
    parser.add_argument("--kmeans_iter", type=int, default=10, help="K-means iterations")
    parser.add_argument("--lambda_weight", type=float, default=1.0, help="Laplacian regularization weight")
    parser.add_argument("--mu_weight", type=float, default=1.0, help="Identity regularization weight")
    args = parser.parse_args()

    if args.input is None:
        args.input = f"../data/dataset/MulDiGraph/output_transactions.txt"
    if args.output is None:
        args.output = f"../data/dataset/Data_after_FE/graph_emb.txt"

    return args

class DifferentiableKMeans:
    def __init__(self, n_clusters, beta, max_iter=10):
        self.n_clusters = n_clusters
        self.beta = beta
        self.max_iter = max_iter

    def initialize_centroids(self, embeddings):
        n_samples = embeddings.shape[0]
        centroids = [embeddings[np.random.randint(n_samples)]]
        for _ in range(self.n_clusters - 1):
            distances = torch.cdist(embeddings, torch.stack(centroids), p=2).min(dim=1)[0]
            probs = distances / distances.sum()
            next_centroid_idx = np.random.choice(n_samples, p=probs.numpy())
            centroids.append(embeddings[next_centroid_idx])
        return torch.stack(centroids)

    def fit(self, embeddings):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        centroids = self.initialize_centroids(embeddings)
        for _ in range(self.max_iter):
            cos_sim = F.cosine_similarity(embeddings.unsqueeze(1), centroids.unsqueeze(0), dim=2)
            assignments = torch.softmax(self.beta * cos_sim, dim=1)
            new_centroids = torch.matmul(assignments.t(), embeddings)
            new_centroids = new_centroids / (assignments.sum(dim=0, keepdim=True).t() + 1e-10)
            centroids = F.normalize(new_centroids, p=2, dim=1)
        cos_sim = F.cosine_similarity(embeddings.unsqueeze(1), centroids.unsqueeze(0), dim=2)
        assignments = torch.softmax(self.beta * cos_sim, dim=1)
        return assignments, centroids

def construct_adjacency_matrix(G, nv, alpha):
    """Construct graph Laplacian matrix with temporal decay."""
    row, col, data = [], [], []
    for v in range(nv):
        for (t, lii, lio) in G[v]:
            weight = np.exp(-t / alpha)
            lii_list = lii.tolist() if lii is not None and lii.size > 0 else []
            lio_list = lio.tolist() if lio is not None and lio.size > 0 else []
            for u in lii_list:
                row.append(v)
                col.append(u)
                data.append(weight)
            for u in lio_list:
                row.append(v)
                col.append(u)
                data.append(weight)
    A = sp.csr_matrix((data, (row, col)), shape=(nv, nv))
    A = A + A.T
    A = (A > 0).astype(float)
    D = sp.diags(A.sum(axis=1).A1)
    L = D - A
    return L, A

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

    # print(f"Computing Betweenness centrality...")
    # features['betweenness'] = nx.betweenness_centrality(A_nx_directed)

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

def construct_cluster_laplacian(A, assignments, n_clusters, nv):
    """Construct cluster-specific Laplacian matrices."""
    cluster_laplacians = []
    for c in range(n_clusters):
        membership = assignments[:, c].reshape(-1, 1)
        M_c = sp.diags(membership.flatten())
        A_c = M_c @ A @ M_c
        A_c = (A_c > 0).astype(float)
        D_c = sp.diags(A_c.sum(axis=1).A1)
        L_c = D_c - A_c
        cluster_laplacians.append(L_c)
    return cluster_laplacians

def laplacian_optimization(embeddings, G, assignments, centroids, alpha, lambda_weight, mu_weight):
    """Optimize embeddings using Laplacian regularization with sparse matrices."""
    nv, d = embeddings.shape
    n_clusters = assignments.shape[1]
    L, A = construct_adjacency_matrix(G, nv, alpha)
    cluster_laplacians = construct_cluster_laplacian(A, assignments, n_clusters, nv)
    L_c_sum = sum(cluster_laplacians)
    I = sp.eye(nv)
    A_matrix = L + lambda_weight * L_c_sum + mu_weight * I
    B_matrix = mu_weight * embeddings   
    Z = np.zeros_like(embeddings)
    for i in range(d):
        Z[:, i], _ = cg(A_matrix, B_matrix[:, i], x0=embeddings[:, i], maxiter=100)
    scaler = MinMaxScaler()
    Z = scaler.fit_transform(Z)
    return Z

def dirtemporalAggregation1(embd, G, v, alpha, freq_features=None, stat_features=None, graph_features=None):
    """Aggregate temporal neighbor embeddings for a single node with all feature combinations."""
    k = embd.shape[1]
    h = np.zeros((k * 2, k * 2))
    h1 = np.zeros((1, k * 2))
    w = []
    
    for i in range(len(G[v])):
        (ti, lii, lio) = G[v][i]
        lii_list = lii.tolist() if lii is not None and lii.size > 0 else []
        lio_list = lio.tolist() if lio is not None and lio.size > 0 else []
        wiin = np.zeros((k,))
        wiout = np.zeros((k,))
        
        for f in lii_list:
            wiin += embd[f, :]
        for g in lio_list:
            wiout += embd[g, :]
            
        wiboth = np.hstack([wiin, wiout])
        wiboth = wiboth / (np.linalg.norm(wiboth) + 1e-10)
        h1 += wiboth
        w.append(wiboth.reshape((k * 2, 1)))
    
    z = np.zeros((1, k * 2))
    for i in range(1, len(G[v])):
        (tni, lii, lio) = G[v][i]
        (tnim1, lim1i, lim1o) = G[v][i - 1]
        exp_input = np.clip((tni - tnim1) / alpha, -20, 20)
        z = np.exp(exp_input) * (w[i - 1].transpose() + z)
        z = z / (np.linalg.norm(z) + 1e-10)
        a = w[i] * z
        h += a
    
    g = h.flatten()
    
    # Base temporal features - this is where you wanted to concat
    temporal_features = np.hstack([g.reshape((1, g.shape[0])), h1])
    
    # Collect all additional features based on feature combination
    additional_features = []
    
    # Add frequency features (6 dimensions)
# Add frequency features (6 dimensions)
    # node_freq_features = np.array([
    #         freq_features[v]['Long-term transfer frequency'], # rank 13 (0.186)
    #         freq_features[v]['Short-term transfer frequency'], # rank 18 (0.102)
    #         freq_features[v]['Long-term incoming transfer frequency'], # rank 14 (0.181)
    #         freq_features[v]['Short-term incoming transfer frequency'], # rank 17 (0.104)
    #         freq_features[v]['Long-term outgoing transfer frequency'], # rank 15 (0.145)
    #         freq_features[v]['Short-term outgoing transfer frequency'] # rank 23 (0.065)
    # ])
    # additional_features.append(node_freq_features)

    # Add statistical features (19 dimensions)
    node_stat_features = np.array([
            # stat_features[v]['node_outdegree'], # rank 16 (0.119)
            stat_features[v]['node_indegree'], # rank 1 (0.738) - highest correlation
            stat_features[v]['direction_ratio'], # rank 2 (0.679)
            stat_features[v]['max_outgoing_amount'], # rank 9 (0.344)
            stat_features[v]['min_outgoing_amount'], # rank 10 (0.344)
            # stat_features[v]['max_incoming_amount'], # rank 21 (0.077)
            # stat_features[v]['min_incoming_amount'], # rank 20 (0.077)
            stat_features[v]['average_outgoing_amount'], # rank 8 (0.344)
            # stat_features[v]['average_incoming_amount'], # rank 19 (0.077)
            stat_features[v]['account_balance'], # rank 7 (0.359)
            # stat_features[v]['account_lifetime'], # rank 22 (0.077)
            # stat_features[v]['active_days'], # rank 12 (0.218)
            # stat_features[v]['mean_hour_sent'], # rank 25 (0.016) - not significant
            stat_features[v]['mean_hour_received'], # rank 4 (0.578)
            stat_features[v]['std_hour_received'], # rank 3 (0.624)
            # stat_features[v]['avg_time_between_tx'], # rank 11 (0.333) - negative correlation
            stat_features[v]['min_time_between_tx'], # rank 5 (0.532) - negative correlation
            # stat_features[v]['max_time_between_tx'], # rank 24 (0.044) - negative correlation
            stat_features[v]['wd_tx_ratio_received'] # rank 6 (0.457)
        ])
    additional_features.append(node_stat_features)
    
    # Add centrality features (7 dimensions)
    # node_centrality_features = np.array([
    #         graph_features['katz'].get(v, 0.0),
    #         # graph_features['betweenness'].get(v, 0.0),
    #         graph_features['degree'].get(v, 0.0),
    #         graph_features['closeness'].get(v, 0.0),
    #         graph_features['clustering'].get(v, 0.0),
    #         graph_features['eigenvector'].get(v, 0.0),
    #         graph_features['indegree'].get(v, 0.0),
    #         graph_features['outdegree'].get(v, 0.0)
    #     ])
    # additional_features.append(node_centrality_features)
    
    # Concatenate all features: [temporal + selected features]
    if additional_features:
        all_additional = np.concatenate(additional_features)
        final_features = np.hstack([temporal_features, all_additional.reshape(1, -1)])
        return final_features
    else:
        return temporal_features

def dirtemporalAggregation(embd, G, alpha, freq_features=None, stat_features=None, graph_features=None):
    """Aggregate temporal neighbor embeddings for all nodes with selected feature combinations."""
    m = []
    nv = len(G)
    for v in range(nv):
        m.append(dirtemporalAggregation1(embd, G, v, alpha, freq_features, stat_features, graph_features))
    return np.vstack(m)

def getnumber(emb):
    """Calculate the number of unique embeddings."""
    ss = set()
    for x in range(emb.shape[0]):
        sd = ','.join(str(emb[x, y]) for y in range(emb.shape[1]))
        ss.add(sd)
    return len(ss)

def dirtemporalSirGN(G, n, alpha, iter=10, beta=10.0, kmeans_iter=10, lambda_weight=1.0, mu_weight=1.0, 
                     freq_loader=None, stat_loader=None):
    """Run SIRGN with Laplacian optimization on assignments and selected feature combinations."""
    nv = len(G)
    embd = np.array([[1 / n for i in range(n)] for x in range(nv)])
    
    # Calculate frequency features if needed
    freq_features = None
    print("Calculating frequency features...")
    freq_features = freq_loader.cal_stat_feats()
    print(f"Frequency features calculated for {len(freq_features)} nodes")
    
    # Calculate statistical features if needed
    stat_features = None
    print("Calculating statistical features...")
    stat_features = stat_loader.cal_stat_feats()
    print(f"Statistical features calculated for {len(stat_features)} nodes")
    
    # Calculate graph features if needed
    graph_features = None
    print("Computing graph centrality features...")
    graph_features = compute_centrality_features(G, nv, alpha)
    
    print(f"Memory before aggregation: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
    emb = dirtemporalAggregation(embd, G, alpha, freq_features, stat_features, graph_features)

    kmeans = DifferentiableKMeans(n_clusters=n, beta=beta, max_iter=kmeans_iter)

    for i in range(iter):
        print(f"Iteration {i}")
        scaler = MinMaxScaler()
        emb1 = scaler.fit_transform(emb)
        print(f"Memory after scaling: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
        emb_torch = torch.from_numpy(emb1).float()
        assignments, centroids = kmeans.fit(emb_torch)
        print(f"Memory after K-means: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

        val = 1 - F.cosine_similarity(emb_torch.unsqueeze(1), centroids.unsqueeze(0), dim=2)
        val = val.detach().numpy()
        M = val.max(axis=1)
        m = val.min(axis=1)
        subx = (M.reshape(nv, 1) - val) / (M - m + 1e-10).reshape(nv, 1)
        su = subx.sum(axis=1) + 1e-10
        subx = subx / su.reshape(nv, 1)

        emb = laplacian_optimization(subx, G, assignments.numpy(), centroids.numpy(), alpha, lambda_weight, mu_weight)
        print(f"Memory after Laplacian: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
        emb = dirtemporalAggregation(emb, G, alpha, freq_features, stat_features, graph_features)
        print(f"Memory after aggregation: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
    return emb

def dirtemporalSirGNStop(G, n, alpha, iter=100, beta=10.0, kmeans_iter=10, lambda_weight=1.0, mu_weight=1.0,
                         freq_loader=None, stat_loader=None):
    """Run SIRGN with early stopping and Laplacian optimization with selected feature combinations."""
    nv = len(G)
    embd = np.array([[1 / n for i in range(n)] for x in range(nv)])
    
    # Calculate frequency features if needed
    freq_features = None
    print("Calculating frequency features...")
    freq_features = freq_loader.cal_stat_feats()
    print(f"Frequency features calculated for {len(freq_features)} nodes")
    
    # Calculate statistical features if needed
    stat_features = None
    print("Calculating statistical features...")
    stat_features = stat_loader.cal_stat_feats()
    print(f"Statistical features calculated for {len(stat_features)} nodes")
    
    # Calculate graph features if needed
    graph_features = None
    print("Computing graph centrality features...")
    graph_features = compute_centrality_features(G, nv, alpha)
    
    print(f"Memory before aggregation: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
    emb = dirtemporalAggregation(embd, G, alpha, freq_features, stat_features, graph_features)
    count = getnumber(emb)
    print('count', count)

    kmeans = DifferentiableKMeans(n_clusters=n, beta=beta, max_iter=kmeans_iter)

    for i in range(iter):
        print(f"Iteration {i}")
        scaler = MinMaxScaler()
        emb1 = scaler.fit_transform(emb)
        print(f"Memory after scaling: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
        emb_torch = torch.from_numpy(emb1).float()
        assignments, centroids = kmeans.fit(emb_torch)
        print(f"Memory after K-means: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

        val = 1 - F.cosine_similarity(emb_torch.unsqueeze(1), centroids.unsqueeze(0), dim=2)
        val = val.detach().numpy()
        M = val.max(axis=1) 
        m = val.min(axis=1)
        subx = (M.reshape(nv, 1) - val) / (M - m + 1e-10).reshape(nv, 1)
        su = subx.sum(axis=1) + 1e-10
        subx = subx / su.reshape(nv, 1)

        emb2 = laplacian_optimization(subx, G, assignments.numpy(), centroids.numpy(), alpha, lambda_weight, mu_weight)
        print(f"Memory after Laplacian: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
        emb2 = dirtemporalAggregation(emb2, G, alpha, freq_features, stat_features, graph_features)
        print(f"Memory after aggregation: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

        count1 = getnumber(emb2)
        print('count', count1)

        if count >= count1:
            print(f"Converged at iteration {i}, unique embeddings: {count1}")
            break
        else:
            emb = emb2
            count = count1
    return emb

def main(args):
    data = pd.read_csv(args.input)
    freq_loader = directed_freq_loader.directed_loader()
    freq_loader.read(data)
    nv = len(freq_loader.G)
    stat_loader = directed_stat_loader.directed_loader()
    stat_loader.read(data)

    edge_count = 0
    for v in range(nv):
        for t, lii, lio in freq_loader.G[v]:
            lii_list = lii.tolist() if lii is not None and lii.size > 0 else []
            lio_list = lio.tolist() if lio is not None and lio.size > 0 else []
            neighbors = lii_list + lio_list
            edge_count += len([(t, u) for u in neighbors])
    edge_count //= 2

    print(edge_count)

    if args.stop:
        emb = dirtemporalSirGNStop(freq_loader.G, args.clusters, args.alpha, args.depth, args.beta, args.kmeans_iter, args.lambda_weight, args.mu_weight,
                                   freq_loader, stat_loader)
    else:
        emb = dirtemporalSirGN(freq_loader.G, args.clusters, args.alpha, args.depth, args.beta, args.kmeans_iter, args.lambda_weight, args.mu_weight,
                               freq_loader, stat_loader)

    freq_loader.storeEmb(args.output, emb)

if __name__ == "__main__":
    args = parse_args()
    main(args)
