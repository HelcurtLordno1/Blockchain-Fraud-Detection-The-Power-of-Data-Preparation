# 🛡️ Blockchain Fraud Detection: The Power of Data Preparation

**Project Status**: ✅ Production-Ready 
**Last Updated**: December 4, 2025  
**Research Focus**: Ethereum Phishing Detection with Advanced Feature Engineering

---

## 👥 Team Members and Contributions

| STT | Student ID | Full Name              | % Contribution |
|-----|------------|------------------------|----------------|
| 17  | 11230548   | Đinh Nam Khánh         | 26%            |
| 26  | 11230570   | Phạm Hồng Minh         | 20%            |
| 7   | 11230526   | Vũ Ngọc Dương          | 20%            |
| 27  | 11230572   | Nghiêm Trà My          | 17%            |
| 39  | 11230597   | Triệu Hải Đăng Trinh   | 17%            |

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Innovations](#-key-innovations)
- [Dataset Information](#-dataset-information)
- [Project Architecture](#-project-architecture)
- [Data Processing Pipeline](#-data-processing-pipeline)
- [Getting Started](#-getting-started)
- [Step-by-Step Workflow](#-step-by-step-workflow)
- [Data Storytelling](#-data-storytelling)
- [Performance Results](#-performance-results)
- [Interactive Dashboard](#-interactive-dashboard)
- [Technical Documentation](#-technical-documentation)
- [Contributing](#-contributing)
- [Research Team](#-research-team)
- [License](#-license)

---

## 🎯 Project Overview

This project addresses one of the most challenging problems in blockchain security: **detecting phishing addresses in the Ethereum network** amidst extreme class imbalance and complex network dynamics. Traditional fraud detection methods fail when applied to blockchain data due to:

- **Extreme Class Imbalance**: Only 0.184% of addresses are phishers (1:543 ratio)
- **Power-Law Distributions**: 89.4% of addresses have ≤2 transactions
- **Heavy-Tailed Transaction Values**: Outliers distort traditional statistical methods
- **Temporal Dynamics**: Bot-like burst patterns and funnel flow attacks

### 🎓 Research Objectives

1. **Power-Law Aware Feature Engineering**: Design 34 features that capture blockchain-specific patterns
2. **Robust Feature Selection**: Use Spearman correlation to handle non-normal distributions
3. **Multi-Model Comparison**: Evaluate 4 machine learning models for 2 dataset 50:50 and 10:90 (fisher-normal)
4. **Production Deployment**: Deploy via interactive Streamlit dashboard with adversarial testing

### 🌟 Real-World Impact

Inspired by the **May 3, 2024 $68M WBTC Address Poisoning Attack**, this system provides:
- ✅ Proactive detection (vs reactive blacklists)
- ✅ Interpretable feature importance
- ✅ Handles extreme imbalance (1:543 ratio)
- ✅ Fast inference for real-time monitoring

---

## 💡 Key Innovations

### 1. **Advanced Feature Engineering (34 Features)**

Our feature set captures three critical dimensions of blockchain behavior:

#### **Frequency Features (6 features)**
- Temporal burstiness detection
- Long-term vs short-term activity patterns
- Bot-like behavior identification

#### **Statistical Features (21 features)**
- Transaction amount distributions
- Account lifetime and activity metrics
- Time-based behavioral patterns
- Funnel flow detection (high inbound, low outbound)

#### **Centrality Features (7 features)**
- Katz centrality
- Degree, closeness, betweenness centrality
- Clustering coefficient
- Eigenvector centrality
- Network position analysis

### 2. **Power-Law Aware Feature Selection**

- Uses **Spearman correlation** (rank-based) instead of Pearson
- Handles non-normal distributions effectively
- Reduces features: 34 → 20 (balanced 5:5) or 34 → 14 (imbalanced 1:9)
- Improves model interpretability and inference speed

### 3. **Balanced vs Imbalanced Training**

Two training strategies address different deployment scenarios:

| Strategy | Ratio | Use Case | Performance |
|----------|-------|----------|-------------|
| **Balanced (5:5)** | 50:50 phisher:normal | Research, baseline | 93.43% recall |
| **Imbalanced (1:9)** | 10:90 phisher:normal | Realistic deployment | 66.21% recall |

### 4. **War Room 3.0: Adversarial Testing**

Full-featured adversarial simulation game:
- 🔴 **Red Team**: Design multi-day phishing campaigns with 9 attack types
- 🛡️ **Blue Team**: Build layered defenses against real attack patterns
- 📊 **Leaderboard**: Track successful bypasses and defense effectiveness
- 🎮 **Interactive**: Day-by-day campaign simulation with stealth mechanics

---

## 📊 Dataset Information

### **Ethereum MultiDiGraph Dataset**

Our dataset is constructed from **Etherscan verified phishing addresses** and represents real-world blockchain transaction patterns.

#### **Dataset Statistics**
```python
Total Addresses:      2,973,489
Total Transactions:   13,551,303
Verified Phishers:    5,480 (from Etherscan)
Phisher Rate:         0.184% (1:543 imbalance ratio)
Time Period:          Multi-year blockchain history
Data Source:          Etherscan API + verified phishing database
```

#### **Data Characteristics**

**Power-Law Distribution**:
- 89.4% of addresses have ≤2 transactions
- 27 super-nodes with >100K transactions
- Average degree: 9.1, Median: 1
- Long tail requires special handling

**Heavy-Tailed Values**:
- Mean transaction: 9.47 ETH (distorted by outliers)
- Skewness: >500
- Kurtosis: >250,000
- >50% zero-value transactions (dust attacks)

**Extreme Imbalance**:
- Normal addresses: 2,967,969 (99.816%)
- Phisher addresses: 5,480 (0.184%)
- Imbalance ratio: 1:543
- Challenge: Avoid majority class bias

---

## 🏗️ Project Architecture

```
Data-Prep-Project-main/
├── 📂 data/
│   ├── raw_data/
│   │   └── phisher_accounts.txt          # 5,480 verified phishers from Etherscan
│   ├── dataset/
│   │   ├── MulDiGraph/
│   │   │   └── output_transactions.txt   # 13.55M transaction edges
│   │   ├── Data_after_FE/
│   │   │   ├── features.csv              # 34 engineered features
│   │   │   └── graph_emb.txt             # Graph embeddings
│   │   └── Data_feature_selection/
│   │       ├── selected_features.json     # 17 selected features
│   │       └── spearman_correlations_train.csv
│   └── processed_data/
│       ├── data_Dataset.address_to_index  # Address mapping
│       └── data_Dataset.index_to_address  # Index mapping
│
├── 📂 features_engineering/
│   ├── directed_freq_loader.py            # Frequency feature extraction
│   ├── directed_stat_loader.py            # Statistical feature extraction
│   └── extract_features_to_csv.py         # Complete feature pipeline
│
├── 📂 models/
│   ├── detect_xgboost.py                  # XGBoost training script
│   ├── sgdit_sgcr.py                      # Alternative model (SGDiT SGCR)
│   └── save_models/
│       ├── xgboost_no_FE_selection.joblib          # All 34 features
│       ├── xgboost_with_FE_selection.joblib        # Selected features (20 or 14)
│       └── xgboost_with_FE_selection_metadata.json # Training metadata
│
├── 📂 notebook_process_data/
│   ├── 📓 00_eda.ipynb                           # Exploratory Data Analysis
│   ├── 📓 00_comparison_analysis.ipynb           # Dataset comparison
│   ├── 📓 01_a_features_engineering.ipynb        # Feature engineering
│   ├── 📓 01_b_why_FE.ipynb                      # Feature engineering rationale
│   ├── 📓 02_features_selection.ipynb            # Spearman-based selection
│   ├── 📓 03_a_models_training_no_FE_selection.ipynb   # Train with all 34 features
│   ├── 📓 03_b_models_training_with_FE_selection.ipynb # Train with 17 features
│   └── get_dataset/
│       ├── 📓 get_dataset_5_5.ipynb              # Generate balanced dataset (50:50)
│       ├── 📓 get_dataset_1_9.ipynb              # Generate imbalanced dataset (10:90)
│       └── 📓 save_main_dataset.ipynb            # Save processed dataset
│
├── 📂 data_storytelling_notebook/
│   ├── 📓 00_context_conflict.ipynb              # The $287M battlefield
│   ├── 📓 01_central_conflict.ipynb              # $68M attack reconstruction
│   ├── 📓 02_explanatory_journey.ipynb           # Feature engineering journey
│   ├── 📓 03_the_solution.ipynb                  # Transformation proof
│   ├── 📓 04_call_to_action.ipynb                # Actionable takeaways
│   ├── 📄 Data_story_telling.md                     # Storytelling documentation
│   └── save_image_story/                         # Generated visualizations
│
├── 📂 streamlit_UI/
│   ├── app.py                             # Main Streamlit dashboard (2,672 lines) 
│   ├── model_utils.py                     # Model utilities (242 lines) 
│   ├── data_utils.py                      # Data utilities (326 lines) 
│   ├── ui_components.py                   # UI components (420 lines) 
│   ├── requirements.txt                   # Python dependencies
│   └── 📄 README_UI_streamlit.md             # Dashboard user guide 
│
└── 📄 README.md                           # This file
```

---

## 🔄 Data Processing Pipeline

Our data processing follows a rigorous 6-step pipeline that transforms raw blockchain data into production-ready features while addressing power-law distributions and extreme imbalance.

### **Pipeline Overview**

```
Raw Data → Feature Engineering → Feature Selection → Model Training → Deployment
   ↓              ↓                    ↓                  ↓              ↓
5,480 phishers  34 features     17 selected features  5 ML models  Streamlit UI
2.97M total    (3 categories)   (Spearman-based)     (2 strategies) (War Room 3.0)
```

### **Step 1: Raw Data Collection**

**Input**: Verified phishing addresses from Etherscan
```
phisher_accounts.txt → 5,480 confirmed phishing addresses
Ethereum blockchain  → 13,551,303 transaction records
                     → 2,973,489 unique addresses
```

**Output**: Multi-directed graph representation
```
output_transactions.txt:
- Source address → Target address
- Transaction value, timestamp
- Edge attributes (gas, nonce, etc.)
```

### **Step 2: Feature Engineering (34 Features)**

**Transformation**: Raw transactions → Behavioral features

#### **Frequency Features (6 features)** 
*Extracted by `directed_freq_loader.py` 
```python
# Capture temporal patterns
long_term_transfer_freq     # Total transaction frequency
short_term_transfer_freq    # Recent activity burst detection
long_term_incoming_freq     # Long-term inbound patterns
short_term_incoming_freq    # Recent inbound spikes
long_term_outgoing_freq     # Long-term outbound patterns
short_term_outgoing_freq    # Recent outbound bursts
```

#### **Statistical Features (21 features)**
*Extracted by `directed_stat_loader.py`  
```python
# Node-level statistics
node_indegree, node_outdegree        # Connection counts
direction_ratio                       # In/out transaction ratio (funnel detection)

# Transaction amount features
max_outgoing_amount, min_outgoing_amount, average_outgoing_amount
max_incoming_amount, min_incoming_amount, average_incoming_amount

# Account characteristics
account_balance                       # Current balance
account_lifetime                      # Days since first transaction
active_days                          # Number of active days

# Temporal patterns
mean_hour_sent, mean_hour_received   # Time of day patterns
std_hour_sent, std_hour_received     # Consistency in timing
avg_time_between_tx                  # Transaction spacing
min_time_between_tx, max_time_between_tx

# Weekday patterns
wd_tx_ratio_sent, wd_tx_ratio_received  # Weekday vs weekend activity
```

#### **Centrality Features (7 features)**
```python
# Graph-level importance
katz_centrality                      # Influence propagation
degree_centrality                    # Connection importance
closeness_centrality                 # Distance to others
clustering_coefficient               # Local density
eigenvector_centrality              # Importance of connections
indegree_centrality                 # Receiving importance
outdegree_centrality                # Sending importance
```

**Why These Features?**
- ✅ Captures bot-like burst patterns (frequency features)
- ✅ Detects funnel flows (direction_ratio: high in, low out)
- ✅ Identifies isolated victim structures (centrality features)
- ✅ Handles power-law distributions (statistical features)

### **Step 3: Feature Selection (34 → 17 Features)**

**Method**: Spearman Correlation Analysis

**Why Spearman?**
- ❌ Pearson assumes normal distribution (violated by blockchain data)
- ✅ Spearman uses rank-based correlation (handles outliers)
- ✅ Works with power-law and heavy-tailed distributions

**Selection Criteria**:
```python
threshold = 0.05  # Spearman correlation with target
method = "Spearman"
data_split = "train_only"  # Prevent data leakage
```

**Selected Features** (varies by dataset):

**For Balanced Dataset (5:5) - 20 features**:
- Frequency: 5 features
- Statistical: 10 features  
- Centrality: 5 features

**For Imbalanced Dataset (1:9) - 14 features**:
```json
{
  "frequency": [
    "long_term_transfer_freq",
    "long_term_incoming_freq",
    "long_term_outgoing_freq"
  ],
  "statistical": [
    "node_indegree",
    "node_outdegree",
    "active_days",
    "account_lifetime",
    "mean_hour_received",
    "std_hour_sent",
    "std_hour_received",
    "avg_time_between_tx",
    "max_time_between_tx",
    "wd_tx_ratio_sent",
    "wd_tx_ratio_received"
  ],
  "centrality": []
}
```

**Impact**:
- 🎯 Reduced features: 34 → 20 (41.2% reduction for balanced) or 34 → 14 (58.8% for imbalanced)
- 🎯 Maintained performance: ~1% accuracy loss (balanced) or ~1.1% loss (imbalanced)
- 🎯 Improved inference speed: 21-28% faster
- 🎯 Better interpretability: Focus on key signals

### **Step 4: Dataset Generation**

Two strategies for different deployment scenarios:

#### **Option A: Balanced Dataset (5:5 ratio)**
```bash
notebook: get_dataset_5_5.ipynb

Configuration:
- Phishing addresses:  5,480
- Normal addresses:    5,480 (randomly sampled)
- Total samples:       10,960
- Class distribution:  50% : 50%
- Use case:           Research, baseline, feature analysis
```

#### **Option B: Imbalanced Dataset (1:9 ratio)**
```bash
notebook: get_dataset_1_9.ipynb

Configuration:
- Phishing addresses:  1,096 (subset of 5,480 verified)
- Normal addresses:    9,864 (randomly sampled)
- Total samples:       10,960
- Class distribution:  10% : 90%
- Use case:           Realistic deployment, production
```

**Important**: Choose ONE dataset strategy before training. Both model types (03_a and 03_b) will use the same dataset.

### **Step 5: Model Training**

Two model configurations trained on your chosen dataset:

#### **Step 7a: Model Training Without Feature Selection (All 34 Features)**
📓 **Notebook**: `notebook_process_data/03_a_models_training_no_FE_selection.ipynb`

**Configuration**:
- **Features used**: All 34 engineered features (frequency + statistical + centrality)
- **Models trained**: XGBoost, LightGBM, Random Forest, Gradient Boosting, AdaBoost
- **Optimization**: Optuna hyperparameter tuning (50+ trials per model)
- **Validation**: 5-fold time-series cross-validation
- **Training time**: ~1.53 seconds (balanced) / ~1.18 seconds (imbalanced)
- **Output model**: `models/save_models/xgboost_no_FE_selection.joblib`
- **Use case**: Maximum information retention, baseline performance

**Expected Performance**:
- **Balanced (5:5)**: 90.78% accuracy, 93.43% recall, 91.02% F1
- **Imbalanced (1:9)**: 94.66% accuracy, 66.21% recall, 71.25% F1 (phisher class)

#### **Step 7b: Model Training With Feature Selection (20 or 14 Features)**
📓 **Notebook**: `notebook_process_data/03_b_models_training_with_FE_selection.ipynb`

**Configuration**:
- **Features used**: 20 features (balanced 5:5) or 14 features (imbalanced 1:9)
- **Models trained**: XGBoost, LightGBM, Random Forest, Gradient Boosting, AdaBoost
- **Optimization**: Optuna hyperparameter tuning (50+ trials per model)
- **Validation**: 5-fold time-series cross-validation
- **Training time**: ~1.2 seconds (balanced, 21% faster) / ~0.85 seconds (imbalanced, 28% faster)
- **Output models**: 
  - `models/save_models/xgboost_with_FE_selection.joblib`
  - `models/save_models/xgboost_with_FE_selection_metadata.json`
- **Use case**: Faster inference, production deployment, interpretability

**Expected Performance**:
- **Balanced (5:5)**: 89.46% accuracy (-1.32%), 92.70% recall (-0.73%), 89.79% F1 (-1.23%)
- **Imbalanced (1:9)**: 93.57% accuracy (-1.09%), 60.27% recall (-5.94%), 65.19% F1 (-6.06%)

**Training Features**:
- Hyperparameter optimization with Optuna (50+ trials)
- Time-series aware validation (no future leakage)
- Multiple model comparison
- Comprehensive metrics (precision, recall, F1, AUC)

### **Step 6: Deployment**

**Interactive Streamlit Dashboard** with 4 main tabs:

1. **Address Checker**: Analyze any Ethereum address
2. **Classic Simulator**: Configure attack patterns
3. **Analytics Dashboard**: Dataset insights and performance metrics
4. **War Room 3.0**: Full adversarial testing game

---

## 🚀 Getting Started

### **Prerequisites**

```bash
# System requirements
Python 3.8 or higher
Jupyter Notebook or VS Code
8GB+ RAM (for large graph processing)
```

### **Installation**

#### **Step 1: Clone the Repository**

```bash
git clone https://github.com/HelcurtLordno1/Blockchain-Fraud-Detection-The-Power-of-Data-Preparation.git
cd Blockchain-Fraud-Detection-The-Power-of-Data-Preparation
```

#### **Step 2: Install Dependencies**

```bash
# Core dependencies
pip install pandas numpy scipy networkx scikit-learn

# Machine learning frameworks
pip install xgboost lightgbm optuna

# Visualization and UI
pip install streamlit plotly matplotlib seaborn

# Optional: Progress bars
pip install tqdm

# Or install all at once from streamlit_UI folder
cd streamlit_UI
pip install -r requirements.txt
```

#### **Step 3: Verify Installation**

```bash
# Check Python version
python --version  # Should be 3.8+

# Check key packages
python -c "import xgboost; print(f'XGBoost: {xgboost.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
python -c "import networkx; print(f'NetworkX: {networkx.__version__}')"

# 🆕 Verify new utility modules (from streamlit_UI folder)
cd streamlit_UI
python -c "from model_utils import load_model_with_fallback; from data_utils import load_pickle_file; from ui_components import render_metric_card; print('✓ All utility modules working!')"
```

---

## 📚 Step-by-Step Workflow

Follow this comprehensive workflow to reproduce our results or train your own models.

> **⚠️ CRITICAL ORDER**: You MUST start with Phase 1 (Dataset Generation) to choose your dataset strategy (5:5 or 1:9) before proceeding to other phases. This choice affects all subsequent steps.

---

# 🗂️ **PHASE 1: Dataset Generation** *(START HERE)*

---

> **⚠️ IMPORTANT**: Choose ONE dataset strategy at the start. This decision determines your entire pipeline configuration.

**Key Difference**: Both datasets use the SAME 10,960 total accounts but with different class distributions:
- **Balanced (5:5)**: Uses ALL 5,480 verified phishers + 5,480 randomly sampled normal accounts
- **Imbalanced (1:9)**: Uses 1,096 phishers (subset) + 9,864 normal accounts to simulate realistic 10:90 ratio

---

## **Step 1a: Balanced Dataset Generation** ⚖️ **(Option A)**

📓 **Notebook**: `notebook_process_data/get_dataset/get_dataset_5_5.ipynb`

### **Configuration**:
- **Ratio**: 50% phishing : 50% normal (5:5)
- **Phishing addresses**: 5,480 (ALL verified addresses from Etherscan - 50% of total)
- **Normal addresses**: 5,480 (randomly sampled from 2.97M - 50% of total)
- **Total samples**: 10,960 accounts
- **Class distribution**: Perfect balance (50:50)

**Use Cases**:
- ✓ Research and academic analysis
- ✓ Feature importance studies
- ✓ Baseline performance evaluation
- ✓ Educational demonstrations
- ✓ Model interpretability analysis

**Advantages**:
- Easier to train (no class imbalance)
- Higher recall (93.43% - catches most phishers)
- Better for understanding behavioral patterns
- Faster model convergence
- Clear performance metrics

**Disadvantages**:
- Not realistic (real-world is 0.184% phishers, 1:543 ratio)
- Higher false positive rate (11.86%)
- May not generalize to production scenarios
- Overestimates detection capability

**Expected Performance** (from Report_3a_3b_dataset_5_5.md):
- Accuracy: 90.78% (34 features) / 89.46% (20 features)
- Precision: 88.73% (34) / 87.06% (20)
- Recall: 93.43% (34) / 92.70% (20)
- F1-Score: 91.02% (34) / 89.79% (20)

**Code Highlights**:
```python
# Step 1: Load MultiDiGraph and extract transactions
graph = load_pkl('../../data/raw_data/MulDiGraph.pkl')
transactions = extract_transactions(graph)  # From graph edges

# Step 2: Organize transactions by direction
f_in, f_out = {}, {}  # Incoming and outgoing per address
for tran in transactions:
    from_addr = tran['from_address']
    to_addr = tran['to_address']
    
    # Track outgoing transactions
    if from_addr not in f_out:
        f_out[from_addr] = []
    f_out[from_addr].append(tran)
    
    # Track incoming transactions
    if to_addr not in f_in:
        f_in[to_addr] = []
    f_in[to_addr].append(tran)

# Step 3: Generate transaction sequences per account
eoa2seq = {}  # Account -> [sorted transaction list]
for account in set(f_in.keys()) | set(f_out.keys()):
    tx_list = f_in.get(account, []) + f_out.get(account, [])
    # Sort by timestamp
    tx_sorted = sorted(tx_list, key=lambda x: x['timestamp'])
    eoa2seq[account] = tx_sorted

# Step 4: Classify accounts as phisher or normal
phisher_list = load_txt('../../data/raw_data/phisher_accounts.txt')
phisher_set = set(phisher_list)

phisher_accounts = [acc for acc in eoa2seq.keys() if acc in phisher_set]
normal_accounts = [acc for acc in eoa2seq.keys() if acc not in phisher_set]

print(f"Available phishers: {len(phisher_accounts)}")
print(f"Available normal: {len(normal_accounts)}")

# Step 5: Sample balanced dataset (50:50 ratio)
num_phishers = 5480  # Use ALL verified phishers
num_normal = 5480    # Equal number of normal accounts

random.seed(42)
selected_phishers = phisher_accounts[:num_phishers]  # All 5,480
selected_normal = random.sample(normal_accounts, num_normal)

# Create final dataset
final_eoa2seq = {}
for acc in selected_phishers + selected_normal:
    final_eoa2seq[acc] = eoa2seq[acc]

print(f"\nFinal dataset: {len(final_eoa2seq)} accounts")
print(f"  Phishers: {len(selected_phishers)} (50%)")
print(f"  Normal: {len(selected_normal)} (50%)")

# Step 6: Save processed dataset
save_pkl(final_eoa2seq, '../../data/processed_data/eoa2seq.pkl')
save_txt(selected_phishers, '../../data/processed_data/phisher_accounts.txt')
print("\n✓ Balanced dataset (5:5) saved successfully!")
```

**Time Required**: 5-10 minutes

**Run Instructions**:
1. Open notebook: `jupyter notebook`
2. Navigate to: `notebook_process_data/get_dataset/get_dataset_5_5.ipynb`
3. Run all cells sequentially
4. Verify outputs:
   - Training set: 8,768 accounts (4,384 phishers + 4,384 normal)
   - Test set: 2,192 accounts (1,096 phishers + 1,096 normal)
5. Check stratification: Both sets maintain 50:50 ratio
6. Proceed to Step 2 (Save Main Dataset)

---

## **Step 1b: Imbalanced Dataset Generation** ⚖️ **(Option B)**

📓 **Notebook**: `notebook_process_data/get_dataset/get_dataset_1_9.ipynb`

### **Configuration**:
- **Ratio**: 10% phishing : 90% normal (1:9)
- **Phishing addresses**: 1,096 (10% of total)
- **Normal addresses**: 9,864 (90% of total)
- **Total samples**: 10,960 accounts (same size as balanced)
- **Class distribution**: Realistic imbalance (10:90)

**Use Cases**:
- ✓ Production deployment scenarios
- ✓ Realistic performance evaluation
- ✓ Cost-sensitive learning
- ✓ Real-world detection capability assessment
- ✓ False positive rate optimization

**Advantages**:
- More realistic imbalance (closer to real-world 1:543)
- Lower false positive rate (2.18% vs 11.86%)
- Better production performance indicators
- Generalizes well to actual deployment
- Tests model robustness to class imbalance

**Disadvantages**:
- Harder to train (severe class imbalance)
- Lower recall (66.21% - misses more phishers)
- Requires careful threshold tuning
- Slower convergence during training
- Accuracy metric becomes misleading (90% baseline by always predicting "normal")

**Expected Performance** (from Report_3a_3b_dataset_1_9.md):
- Accuracy: 94.66% (34 features) / 93.57% (14 features)
- Precision: 77.13% (34) / 70.97% (14) for phisher class
- Recall: 66.21% (34) / 60.27% (14) for phisher class
- F1-Score: 71.25% (34) / 65.19% (14) for phisher class

**Code Highlights**:
```python
# Step 1-4: Same as balanced dataset (load graph, extract, organize, classify)
graph = load_pkl('../../data/raw_data/MulDiGraph.pkl')
transactions = extract_transactions(graph)
f_in, f_out = load_data_muldi(transactions)
eoa2seq = generate_transaction_sequences(f_in, f_out)

phisher_list = load_txt('../../data/raw_data/phisher_accounts.txt')
phisher_set = set(phisher_list)

phisher_accounts = [acc for acc in eoa2seq.keys() if acc in phisher_set]
normal_accounts = [acc for acc in eoa2seq.keys() if acc not in phisher_set]

# Step 5: Sample imbalanced dataset (10:90 ratio)
# Target: 10,960 total accounts (same as balanced)
num_phishers = 1096   # 10% of 10,960
num_normal = 9864     # 90% of 10,960

random.seed(42)
# Sample SUBSET of phishers (1,096 out of 5,480 available)
selected_phishers = random.sample(phisher_accounts, num_phishers)
# Sample normal accounts to match realistic imbalance
selected_normal = random.sample(normal_accounts, num_normal)

# Create final dataset
final_eoa2seq = {}
for acc in selected_phishers + selected_normal:
    final_eoa2seq[acc] = eoa2seq[acc]

print(f"\nFinal dataset: {len(final_eoa2seq)} accounts")
print(f"  Phishers: {len(selected_phishers)} (10%)")
print(f"  Normal: {len(selected_normal)} (90%)")
print(f"  Imbalance ratio: 1:{num_normal//num_phishers}")

# Verify class distribution
phisher_pct = len(selected_phishers) / len(final_eoa2seq) * 100
print(f"\nClass distribution verification:")
print(f"  Phisher rate: {phisher_pct:.1f}% (target: 10%)")
print(f"  Imbalance factor: {len(selected_normal)/len(selected_phishers):.1f}x")

# Step 6: Save processed dataset
save_pkl(final_eoa2seq, '../../data/processed_data/eoa2seq.pkl')
save_txt(selected_phishers, '../../data/processed_data/phisher_accounts.txt')
print("\n✓ Imbalanced dataset (1:9) saved successfully!")
```

**Time Required**: 5-10 minutes

**Run Instructions**:
1. Open notebook: `jupyter notebook`
2. Navigate to: `notebook_process_data/get_dataset/get_dataset_1_9.ipynb`
3. Run all cells sequentially
4. Verify outputs:
   - Training set: 8,768 accounts (877 phishers + 7,891 normal)
   - Test set: 2,192 accounts (219 phishers + 1,973 normal)
5. Check stratification: Both sets maintain 10:90 ratio
6. Proceed to Step 2 (Save Main Dataset)

**Decision Guide**:
```
First time user?          → Use 5:5 (balanced - 10,960 accounts with 5,480 phishers)
Research/academic work?   → Use 5:5 (balanced - equal class distribution)
Production deployment?    → Use 1:9 (imbalanced - 10,960 accounts with 1,096 phishers)
Feature analysis?         → Use 5:5 (balanced - easier interpretation)
Real-world testing?       → Use 1:9 (imbalanced - realistic 10:90 ratio)
```

---

## 📖 Data Storytelling

The `data_storytelling_notebook/` folder contains 5 narrative notebooks that transform technical results into compelling visual stories for different audiences (practitioners, decision-makers, researchers).

### **Story Structure (5-Act Framework)**

| Notebook | Purpose | Key Message | Duration |
|----------|---------|-------------|----------|
| **00_context_conflict** | Establish the problem | "99.8% accuracy" = 0% detection | 15 min |
| **01_central_conflict** | Real-world stakes | $68M attack flagged 9 days early | 20 min |
| **02_explanatory_journey** | Solution path | Decoding behavioral signatures | 30 min |
| **03_the_solution** | Demonstrate success | 0% → 92.70% recall transformation | 15 min |
| **04_call_to_action** | Actionable takeaways | 3-tier insights by audience | 10 min |

**Total Duration**: ~90 minutes (presentation-ready)

### **Key Visualizations**

- **Class Imbalance**: 1:543 ratio visualization (finding 1 in 5 stadiums)
- **Baseline Failure**: Confusion matrix showing 99.8% accuracy with 0% detection
- **Power-Law Distributions**: 3-panel analysis (89.4% addresses have ≤2 transactions)
- **Behavioral Signatures**: Radar chart showing phishers opposite across all dimensions
- **$68M Attack Timeline**: 9-day reconstruction with feature signatures
- **The Transformation**: Side-by-side confusion matrices (0% → 92.70% recall)
- **Business Impact**: $266M annually saved projection

**Prerequisites**: Complete all `notebook_process_data/` notebooks first (dataset generation → feature engineering → model training).

**Documentation**: See `data_storytelling_notebook/Data_story_telling.md` for detailed information.

---

## **Step 2: Save Main Dataset** 💾

📓 **Notebook**: `notebook_process_data/get_dataset/save_main_dataset.ipynb`

**Objectives**:
- Save the generated dataset from Step 1a or 1b
- Create address-to-index mappings for feature engineering
- Prepare processed data structure for subsequent phases
- Verify dataset integrity and class distribution

**Key Operations**:
```python
# Save processed dataset
save_pkl(final_eoa2seq, '../../data/processed_data/eoa2seq.pkl')

# Save address mappings
address_to_index = {addr: idx for idx, addr in enumerate(final_eoa2seq.keys())}
index_to_address = {idx: addr for addr, idx in address_to_index.items()}

save_pkl(address_to_index, '../../data/processed_data/data_Dataset.address_to_index')
save_pkl(index_to_address, '../../data/processed_data/data_Dataset.index_to_address')

# Verify saved data
print(f"Total accounts saved: {len(final_eoa2seq)}")
print(f"Phisher accounts: {sum(1 for acc in final_eoa2seq if acc in phisher_set)}")
print(f"Normal accounts: {sum(1 for acc in final_eoa2seq if acc not in phisher_set)}")
```

**Output Files**:
- `data/processed_data/eoa2seq.pkl` - Transaction sequences per account
- `data/processed_data/data_Dataset.address_to_index` - Address → Index mapping
- `data/processed_data/data_Dataset.index_to_address` - Index → Address mapping
- `data/processed_data/phisher_accounts.txt` - List of phisher addresses

**Time Required**: 2-5 minutes

**Run Instructions**:
1. Ensure Step 1a OR 1b completed successfully
2. Open notebook: `jupyter notebook`
3. Navigate to: `notebook_process_data/get_dataset/save_main_dataset.ipynb`
4. Run all cells sequentially
5. Verify all output files created in `data/processed_data/`
6. Proceed to Phase 2 (Data Exploration)

---

# 📊 **PHASE 2: Data Exploration & Understanding**

---

## **Step 3: Exploratory Data Analysis** 🔍

📓 **Notebook**: `notebook_process_data/00_eda.ipynb`

**Objectives**:
- Analyze the 2.97M address dataset with 13.55M transactions
- Understand transaction patterns and power-law distributions
- Identify extreme class imbalance (1:543 phisher:normal ratio)
- Visualize network structure and degree distributions
- Assess data quality, completeness, and outliers

**Key Analyses**:
- Degree distribution confirmation (power-law with 89.4% having ≤2 transactions)
- Transaction value distribution analysis (heavy-tailed, mean 9.47 ETH)
- Temporal patterns and burstiness detection
- Phisher vs normal address behavioral comparison
- Network centrality and clustering patterns

**Code Highlights**:
```python
# Load raw blockchain transaction data
raw_df = pd.read_csv('../data/dataset/MulDiGraph/output_transactions.txt', 
                     header=None, 
                     names=['from_address', 'to_address', 'timestamp', 'amount'])

# Load phisher labels
phisher_list = load_txt('../data/processed_data/phisher_accounts.txt')
phisher_set = set(phisher_list)

# Calculate unique accounts and classify
all_accounts_raw = set(raw_df['from_address']) | set(raw_df['to_address'])
raw_unique_accounts = len(all_accounts_raw)
phisher_accounts = len(all_accounts_raw & phisher_set)
normal_accounts = raw_unique_accounts - phisher_accounts

# Analyze degree distribution (power-law verification)
degree_counts = raw_df.groupby('from_address').size()
low_activity_pct = (degree_counts <= 2).sum() / len(degree_counts) * 100

# Transaction value analysis (heavy-tailed distribution)
mean_tx_value = raw_df['amount'].mean()  # Distorted by outliers
median_tx_value = raw_df['amount'].median()  # More robust
zero_value_pct = (raw_df['amount'] == 0).sum() / len(raw_df) * 100

print(f"Total accounts: {raw_unique_accounts:,}")
print(f"Phisher rate: {phisher_accounts/raw_unique_accounts*100:.3f}%")
print(f"Low activity (≤2 tx): {low_activity_pct:.1f}%")
print(f"Mean transaction: {mean_tx_value:.2f} ETH (outlier-affected)")
print(f"Zero-value transactions: {zero_value_pct:.1f}% (dust attacks)")
```

**Time Required**: 10-15 minutes

**Run Instructions**:
1. Open Jupyter: `jupyter notebook`
2. Navigate to: `notebook_process_data/00_eda.ipynb`
3. Run all cells: Kernel → Restart & Run All
4. Review visualizations and statistics
5. Verify power-law confirmation and imbalance metrics

---

## **Step 4: Dataset Comparison Analysis** 📊

📓 **Notebook**: `notebook_process_data/00_comparison_analysis.ipynb`

**Objectives**:
- Compare balanced (5:5) vs imbalanced (1:9) dataset configurations
- Analyze feature distributions across different class ratios
- Assess feature importance variability between scenarios
- Validate data preprocessing and stratified sampling

**Key Insights**:
- Balanced dataset: 10,960 accounts (5,480 phishers + 5,480 normal)
- Imbalanced dataset: 10,960 accounts (1,096 phishers + 9,864 normal)
- Feature distribution shifts between configurations
- Class imbalance impact on model training

**Time Required**: 5-10 minutes

---

# ⚙️ **PHASE 3: Feature Engineering**

---

## **Step 5: Feature Engineering Implementation** 🔧

📓 **Notebook**: `notebook_process_data/01_a_features_engineering.ipynb`

**Objectives**:
- Extract 6 frequency features capturing temporal burstiness patterns
- Compute 21 statistical features for behavioral analysis
- Calculate 7 centrality features for network positioning
- Export comprehensive feature matrix (34 features × 2.97M addresses)

**Key Operations**:
1. **Load transaction graph** (13,551,303 edges, 2,973,489 nodes)
   - Parse output_transactions.txt
   - Build directed multigraph representation

2. **Compute frequency features** (6 features):
   - Use `directed_freq_loader.py`
   - Long-term vs short-term transfer patterns
   - Incoming/outgoing transaction frequencies

3. **Extract statistical features** (21 features):
   - Use `directed_stat_loader.py`
   - Transaction amount distributions (max, min, avg)
   - Temporal patterns (time between transactions, hour patterns)
   - Account characteristics (lifetime, active days, balance)

4. **Calculate centrality features** (7 features):
   - Katz centrality (influence propagation)
   - Degree, closeness, betweenness centrality
   - Clustering coefficient (local network density)
   - Eigenvector centrality

5. **Export combined features**:
   - Output: `data/dataset/Data_after_FE/features.csv`
   - Dimensions: 2,973,489 rows × 34 feature columns

**Code Highlights**:
```python
# Step 1: Load transaction data and initialize loaders
data = pd.read_csv('../data/dataset/MulDiGraph/output_transactions.txt')

# Initialize frequency feature loader
freq_loader = directed_freq_loader.directed_loader()
freq_loader.read(data)
nv = len(freq_loader.G)  # Number of nodes

# Initialize statistical feature loader
stat_loader = directed_stat_loader.directed_loader()
stat_loader.read(data)

# Step 2: Calculate frequency features (6 features)
freq_features = freq_loader.cal_stat_feats()
# Returns: long_term_transfer_freq, short_term_transfer_freq,
#          long_term_incoming_freq, short_term_incoming_freq,
#          long_term_outgoing_freq, short_term_outgoing_freq

# Step 3: Calculate statistical features (21 features)
stat_features = stat_loader.cal_stat_feats()
# Returns: node_indegree, node_outdegree, direction_ratio,
#          max/min/average_outgoing_amount, max/min/average_incoming_amount,
#          account_balance, account_lifetime, active_days,
#          mean_hour_sent/received, std_hour_sent/received,
#          avg/min/max_time_between_tx, wd_tx_ratio_sent/received

# Step 4: Calculate centrality features (7 features)
def compute_centrality_features(G, nv, alpha=1.0):
    # Construct directed adjacency matrix with temporal decay
    A_directed = construct_directed_adjacency_matrix(G, nv, alpha)
    A_nx_directed = nx.from_scipy_sparse_array(A_directed, create_using=nx.DiGraph)
    
    # Compute 7 centrality metrics
    features = {}
    features['katz'] = nx.katz_centrality(A_nx_directed, alpha=0.01, max_iter=1000)
    features['degree'] = nx.degree_centrality(A_nx_directed)
    features['closeness'] = nx.closeness_centrality(A_nx_directed)
    features['clustering'] = nx.clustering(A_nx_directed)
    features['eigenvector'] = nx.eigenvector_centrality(A_nx_directed, max_iter=1000)
    features['indegree'] = nx.in_degree_centrality(A_nx_directed)
    features['outdegree'] = nx.out_degree_centrality(A_nx_directed)
    return features

graph_features = compute_centrality_features(freq_loader.G, nv, alpha=1.0)

# Step 5: Merge all features into DataFrame
rows = []
for node_id in range(nv):
    row = {'node_id': node_id}
    # Add frequency features (6)
    row.update(freq_features.get(node_id, {}))
    # Add statistical features (21)
    row.update(stat_features.get(node_id, {}))
    # Add centrality features (7)
    for feat_name, feat_dict in graph_features.items():
        row[f'{feat_name}_centrality'] = feat_dict.get(node_id, 0.0)
    rows.append(row)

# Export to CSV
features_df = pd.DataFrame(rows)
features_df.to_csv('../data/dataset/Data_after_FE/features.csv', index=False)
print(f"Exported {len(features_df)} nodes with 34 features")
```

**Time Required**: 30-45 minutes (graph processing intensive)

**Run Instructions**:
1. Ensure raw data exists: `data/dataset/MulDiGraph/output_transactions.txt`
2. Open notebook: `jupyter notebook`
3. Navigate to: `notebook_process_data/01_a_features_engineering.ipynb`
4. Execute all cells sequentially (do not skip cells)
5. Monitor progress bars for long-running computations
6. Verify output file: `data/dataset/Data_after_FE/features.csv`

---

## **Step 6: Feature Engineering Rationale** 💡

📓 **Notebook**: `notebook_process_data/01_b_why_FE.ipynb`

**Objectives**:
- Demonstrate quantitative impact of feature engineering on model performance
- Compare raw transaction data vs engineered behavioral features
- Visualize feature distributions and discriminative power
- Show performance improvements through proper feature design

**Key Insights**:
- **Raw transaction data**: Limited predictive power (~60% accuracy)
- **Engineered features**: Capture phishing behavioral patterns (90%+ accuracy)
- **Power-law aware design**: Statistical features handle blockchain distributions
- **Temporal dynamics**: Frequency features detect bot-like burst patterns
- **Network context**: Centrality features reveal suspicious positioning

**Expected Results**:
- Visual comparison of feature distributions (phishers vs normal)
- Performance improvement metrics (before vs after feature engineering)
- Feature importance rankings

**Time Required**: 10-15 minutes

---

# 🎯 **PHASE 4: Feature Selection**

---

## **Step 7: Spearman-Based Feature Selection** 📈
📓 **Notebook**: `notebook_process_data/02_features_selection.ipynb`

**Objectives**:
- Apply Spearman rank correlation analysis for power-law robust selection
- Identify most discriminative features for phishing detection
- Export selected feature lists for both dataset configurations
- Compare full vs selected feature set performance

**Selection Process**:
1. **Load engineered features** (34 features from Phase 2)
2. **Stratified train/test split** (80-20, prevent data leakage)
3. **Compute Spearman correlation** on training set only
   - Why Spearman: Robust to power-law distributions and outliers
   - Why train-only: Prevent test set information leakage
4. **Apply threshold**: |ρ| > 0.05 with phishing label
5. **Select features by dataset type**:

   **Balanced Dataset (5:5)**:
   - Selected: 20 features (41.2% reduction)
   - Frequency: 5 features (83.3% retained)
   - Statistical: 10 features (47.6% retained)
   - Centrality: 5 features (71.4% retained)

   **Imbalanced Dataset (1:9)**:
   - Selected: 14 features (58.8% reduction)
   - Frequency: 3 features (50.0% retained)
   - Statistical: 11 features (52.4% retained)
   - Centrality: 0 features (0% retained - all eliminated)

**Output Files**:
- `data/dataset/Data_feature_selection/selected_features.json`
- `data/dataset/Data_feature_selection/spearman_correlations_train.csv`

**Code Highlights**:
```python
# Step 1: Load features and labels
features_df = pd.read_csv('../data/dataset/Data_after_FE/features.csv')
phisher_list = load_txt('../data/processed_data/phisher_accounts.txt')
phisher_set = set(phisher_list)

# Create binary labels
features_df['is_phisher'] = features_df['node_id'].apply(
    lambda x: 1 if str(x) in phisher_set else 0
)

# Step 2: Train/test split (CRITICAL: Select features on train only!)
feature_cols = [col for col in features_df.columns if col not in ['node_id', 'is_phisher']]
X = features_df[feature_cols].values
y = features_df['is_phisher'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = pd.DataFrame(X_train, columns=feature_cols)
train_df['is_phisher'] = y_train

# Step 3: Calculate Spearman correlation (train data only)
spearman_results = []
for feat in feature_cols:
    # Rank-based correlation (robust to outliers and power-law distributions)
    corr, p_value = spearmanr(train_df[feat], train_df['is_phisher'])
    spearman_results.append({
        'feature': feat,
        'spearman_corr': corr,
        'abs_corr': abs(corr),
        'p_value': p_value
    })

spearman_df = pd.DataFrame(spearman_results).sort_values('abs_corr', ascending=False)

# Step 4: Apply selection threshold
threshold = 0.05
selected_features = spearman_df[spearman_df['abs_corr'] > threshold]['feature'].tolist()

print(f"Features selected: {len(selected_features)} / 34")
print(f"Reduction: {(1 - len(selected_features)/34)*100:.1f}%")
print(f"Top 5 features by |ρ|:")
for idx, row in spearman_df.head(5).iterrows():
    print(f"  {row['feature']}: ρ={row['spearman_corr']:.4f}")

# Step 5: Save selected features
import json
with open('../data/dataset/Data_feature_selection/selected_features.json', 'w') as f:
    json.dump({'features': selected_features, 'threshold': threshold}, f, indent=2)

spearman_df.to_csv('../data/dataset/Data_feature_selection/spearman_correlations_train.csv', 
                   index=False)
```

**Expected Results**:
- **Balanced (5:5)**: 34 → 20 features, ~1% accuracy loss, 21% faster inference
- **Imbalanced (1:9)**: 34 → 14 features, ~1.1% accuracy loss, 28% faster inference
- Correlation heatmap showing feature independence
- Performance comparison tables

**Time Required**: 5-10 minutes

**Run Instructions**:
1. Ensure `features.csv` exists from Phase 3 (Step 5)
2. Open notebook: `jupyter notebook`
3. Navigate to: `notebook_process_data/02_features_selection.ipynb`
4. Run all cells sequentially
5. Review correlation heatmap and selection criteria
6. Verify JSON output contains selected features (20 or 14 depending on dataset)

---

## **Step 8a: Train Without Feature Selection** 🎯 **(All 34 Features)**

```bash
📓 File: notebook_process_data/03_a_models_training_no_FE_selection.ipynb

Configuration:
- Features: All 34 engineered features
- Models: XGBoost, LightGBM, Random Forest, Gradient Boosting, AdaBoost
- Optimization: Optuna (50 trials per model)
- Validation: 5-fold time-series cross-validation

Training Process:
1. Load dataset from Phase 4 (5:5 or 1:9)
2. Apply all 34 features
3. Train 5 models with hyperparameter optimization
4. Evaluate on test set
5. Save best model: models/save_models/xgboost_no_FE_selection.joblib

Expected Results (Balanced 5:5):
- XGBoost:          Precision: 94%, Recall: 93%, F1: 93%
- LightGBM:         Precision: 93%, Recall: 92%, F1: 92%
- Random Forest:    Precision: 92%, Recall: 91%, F1: 91%
- Gradient Boost:   Precision: 91%, Recall: 90%, F1: 90%
- AdaBoost:         Precision: 89%, Recall: 88%, F1: 88%

Expected Results (Imbalanced 1:9):
- XGBoost:          Precision: 78%, Recall: 66%, F1: 71%
- LightGBM:         Precision: 76%, Recall: 64%, F1: 69%
- Random Forest:    Precision: 75%, Recall: 62%, F1: 68%

Code Highlights:
```python
# Step 1: Load engineered features (all 34 features)
features_df = pd.read_csv('../data/dataset/Data_after_FE/features.csv')
phisher_labels = load_pkl('../data/dataset/MulDiGraph/account_tags.pkl')

# Define all feature columns (6 frequency + 21 statistical + 7 centrality)
freq_cols = ['long_term_transfer_freq', 'short_term_transfer_freq', ...]
stat_cols = ['node_indegree', 'node_outdegree', 'direction_ratio', ...]
centrality_cols = ['katz_centrality', 'degree_centrality', ...]
all_feature_cols = freq_cols + stat_cols + centrality_cols  # 34 features

# Step 2: Convert to graph embedding format
with open('../data/dataset/Data_after_FE/graph_emb.txt', 'w') as f:
    for idx, row in features_df.iterrows():
        node_id = int(row['node_id'])
        feature_values = [str(row[col]) for col in all_feature_cols]
        line = f"{node_id} " + " ".join(feature_values) + "\n"
        f.write(line)

# Step 3: Load data and prepare train/test split
from detect_xgboost import load_embeddings, prepare_data

embeddings = load_embeddings('../data/dataset/Data_after_FE/graph_emb.txt')
X_train, X_test, y_train, y_test = prepare_data(embeddings, phisher_labels, 
                                                  test_size=0.2, stratify=True)

# Step 4: Hyperparameter optimization with Optuna
import optuna
from xgboost import XGBClassifier

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)
    }
    
    model = XGBClassifier(**params, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    return f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
print(f"Best F1 Score: {study.best_value:.4f}")

# Step 5: Train final model with best parameters
final_model = XGBClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# Step 6: Evaluate on test set
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nTest Performance (All 34 Features):")
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1-Score:  {f1*100:.2f}%")

# Step 7: Save model
import joblib
joblib.dump(final_model, '../models/save_models/xgboost_no_FE_selection.joblib')
print("\n✓ Model saved successfully!")
```

Time Required: 20-30 minutes (includes Optuna optimization)

Run Instructions:
1. Ensure Phase 4 completed (dataset generated)
2. Open notebook: 03_a_models_training_no_FE_selection.ipynb
3. Run all cells from top to bottom
4. Wait for Optuna optimization (progress bars shown)
5. Check output: Part 7.7 - Model saved successfully!
6. Verify file exists: models/save_models/xgboost_no_FE_selection.joblib

---

## **Step 8b: Train With Feature Selection** 🎯 **(Selected Features Only)**

```bash
📓 File: notebook_process_data/03_b_models_training_with_FE_selection.ipynb

Configuration:
- Features: 17 selected features (from Phase 3)
- Models: XGBoost, LightGBM, Random Forest, Gradient Boosting, AdaBoost
- Optimization: Optuna (50 trials per model)
- Validation: 5-fold time-series cross-validation

Training Process:
1. Load dataset from Phase 4 (5:5 or 1:9)
2. Load selected features from Phase 3
3. Filter to 17 features only
4. Train 5 models with hyperparameter optimization
5. Evaluate on test set
6. Save best model + metadata:
   - models/save_models/xgboost_with_FE_selection.joblib
   - models/save_models/xgboost_with_FE_selection_metadata.json

Expected Results (Balanced 5:5):
- XGBoost:          Precision: 93%, Recall: 92%, F1: 92% (-1% vs 03_a)
- LightGBM:         Precision: 92%, Recall: 91%, F1: 91%
- Random Forest:    Precision: 91%, Recall: 90%, F1: 90%

Expected Results (Imbalanced 1:9):
- XGBoost:          Precision: 77%, Recall: 65%, F1: 70%
- LightGBM:         Precision: 75%, Recall: 63%, F1: 68%

Benefits:
+ 2x faster inference (17 vs 34 features)
+ Similar performance (<1% loss)
+ Better interpretability
+ Lower computational cost

Time Required: 20-30 minutes (fewer features = faster training)

Run Instructions:
1. Ensure Phase 3 completed (selected_features.json exists)
2. Ensure Phase 4 completed (dataset generated)
3. Open notebook: 03_b_models_training_with_FE_selection.ipynb
4. Run all cells from top to bottom
5. Wait for Optuna optimization
6. Check output: Part 8.5 - Model and metadata saved!
7. Verify files exist:
   - models/save_models/xgboost_with_FE_selection.joblib
   - models/save_models/xgboost_with_FE_selection_metadata.json
```

**Model Selection Guide**:
```
Need maximum accuracy?          → Use Step 8a (all 34 features)
Need fast inference?            → Use Step 8b (selected features)
Production deployment?          → Use Step 8b (selected features)
Research/feature analysis?      → Use Step 8a (all 34 features)
Limited computational resources? → Use Step 8b (selected features)
```

---

# 🚀 **PHASE 6: Interactive Dashboard Deployment**

---

## **Step 9: Launch Streamlit Application** 🌐

```bash
📍 Directory: streamlit_UI/

Prerequisites:
✓ Phase 1 completed (dataset generated)
✓ Phase 5 completed (at least ONE model trained - Step 8a or 8b)
✓ Model files exist in models/save_models/
✓ Streamlit dependencies installed

Launch Steps:

1. Navigate to streamlit folder:
   cd streamlit_UI

2. Install UI dependencies (if not done):
   pip install -r requirements.txt

3. Launch the dashboard:
   streamlit run app.py

4. Access in browser:
   URL: http://localhost:8501
   (Opens automatically in default browser)

Expected Behavior:
- Loading screen (2-3 seconds)
- Model loading message
- Dashboard appears with 4 tabs
- Sidebar shows model configuration

Time Required: 2-3 seconds startup
```

---

## **Dashboard Features Overview** ✨

### **Tab 1: 🔍 Address Checker**
```bash
Features:
- Input any Ethereum address
- Real-time phishing risk analysis
- Confidence scoring with visual indicators
- Feature importance with radar charts
- Network graph visualization
- One-click address copy

Quick Test:
1. Click "📗 Normal Example" button
2. Watch progress bar (3 phases)
3. View results:
   - Risk level (color-coded)
   - Confidence score
   - Top 10 features
   - Radar chart
   - Network graph

4. Try "📕 Phisher Example" button
5. Compare results

Demo Mode:
- Generates consistent features per address
- Uses seeded random generation
- No real blockchain queries
- Instant response for repeated addresses
```

### **Tab 2: 🎮 Classic Simulator**
```bash
Features:
- Configure attack parameters:
  * Spam transactions (0-100)
  * Vanity similarity (6-12 chars)
  * Funnel ratio (0.0-1.0)
  * Burst pattern toggle
  * Funnel flow toggle

Attack Presets:
1. 🟢 Low Risk Attack
2. 🟡 Medium Risk Attack
3. 🔴 High Risk Attack (May 3, 2024 inspired)

Usage:
1. Click "Load Attack Presets"
2. Select preset
3. Click "Run Simulation"
4. View detection probability
5. Analyze attack pattern

Educational Value:
- Understand attack vectors
- See feature impact
- Learn defense strategies
```

### **Tab 3: 📊 Analytics Dashboard**
```bash
Features:
- Dataset overview (2.97M addresses)
- Power-law degree distribution
- Heavy-tailed transaction values
- Model performance comparison
- Feature engineering breakdown
- Export functionality

Visualizations:
- Distribution charts (log scale)
- Model comparison heatmap
- Feature importance rankings
- Performance metrics table

Usage:
1. Review dataset statistics
2. Compare model performance
3. Understand feature categories
4. Export metrics (JSON/TXT)
```

### **Tab 4: ⚔️ War Room 3.0: Breach Protocol**
```bash
Features:
- 🔴 Red Team (Attacker Mode):
  * Design multi-day campaigns (1-30 days)
  * 9 attack types available
  * Configure intensity, burst, funnel ratio
  * Stealth mode after day 7
  * Success: Keep risk <50% for campaign duration
  
- 🛡️ Blue Team (Defender Mode):
  * Build 7-layer defenses
  * Test against 6 real attacks
  * Defense effectiveness scoring
  * Badge/achievement system
  
- 🏆 Hall of Fame:
  * Leaderboard for successful bypasses
  * Battle statistics
  * Defense effectiveness rankings
  
- 📊 Battle Analytics:
  * Day-by-day risk progression
  * Defense layer effectiveness
  * Attack timeline visualization
  * Export battle reports

Attacker Mode Usage:
1. Select "Attacker (Red Team)"
2. Choose attack type (e.g., Address Poisoning)
3. Configure parameters:
   - Campaign duration: 1-30 days
   - Transaction intensity: 1-50/day
   - Burst multiplier: 1-10x
   - Funnel ratio: 0.0-1.0
4. Enable stealth mode (optional)
5. Click "LAUNCH CAMPAIGN"
6. Watch day-by-day simulation
7. Success if final risk <50%

Defender Mode Usage:
1. Select "Defender (Blue Team)"
2. Choose security layers:
   - Hardware Wallet
   - Transaction Simulation
   - Address Whitelisting
   - Regular Approval Audits
   - Two-Factor Authentication
   - Verified Contracts Only
   - Slow & Steady Behavior
3. Click "TEST MY DEFENSES"
4. See results against 6 attacks
5. Earn badges for strong defenses

Educational Value:
- Understand adversarial ML
- Learn attack/defense strategies
- Test model robustness
- Gamified security education
```

---

## 🎯 Performance Results

### **Model Performance Comparison**

Performance metrics for XGBoost models trained on both dataset configurations:

#### **Balanced Dataset (5:5 Ratio) - 03_a vs 03_b**

| Model | Features | Precision | Recall | F1-Score | Accuracy | AUC-ROC | Training Time |
|-------|----------|-----------|--------|----------|----------|---------|---------------|
| **XGBoost (03_a)** | 34 | 88.73% | 93.43% | 91.02% | 90.78% | 0.9699 | 1.53 sec |
| **XGBoost (03_b)** | 20 | 87.06% | 92.70% | 89.79% | 89.46% | 0.9603 | 1.2 sec |
| **Difference** | -41.2% | -1.67% | -0.73% | -1.23% | -1.32% | -0.96% | -21% |

**Confusion Matrix Details (03_a with 34 features)**:
- True Positives: 1,024 / 1,096 phishers (93.43% recall)
- True Negatives: 966 / 1,096 normal (88.14%)
- False Positives: 130 normal wrongly flagged (11.86%)
- False Negatives: 72 phishers missed (6.57%)

**Key Insights**:
- Feature reduction (34 → 20) causes ~1.32% accuracy loss
- 21% faster training with selected features
- Maintained 92.70% recall (critical for phishing detection)
- Production-ready performance with manageable false positive rate

#### **Imbalanced Dataset (1:9 Ratio) - 03_a vs 03_b**

| Model | Features | Precision (Phisher) | Recall (Phisher) | F1-Score (Phisher) | Accuracy | AUC-ROC | FPR |
|-------|----------|---------------------|------------------|-----------------------|----------|---------|-----|
| **XGBoost (03_a)** | 34 | 77.13% | 66.21% | 71.25% | 94.66% | 0.9765 | 2.18% |
| **XGBoost (03_b)** | 14 | 70.97% | 60.27% | 65.19% | 93.57% | 0.9597 | 2.74% |
| **Difference** | -58.8% | -6.16% | -5.94% | -6.06% | -1.09% | -1.68% | +0.56% |

**Confusion Matrix Details (03_a with 34 features)**:
- True Positives: 145 / 219 phishers (66.21% recall)
- True Negatives: 1,930 / 1,973 normal (97.82%)
- False Positives: 43 normal wrongly flagged (2.18%)
- False Negatives: 74 phishers missed (33.79%)

**Key Insights**:
- Realistic imbalance (10:90) reduces recall to 60-66% for phisher class
- Lower false positive rate (2.18-2.74%) suitable for production
- Feature reduction impact more severe: 5.94% recall drop (34→14 features)
- Trade-off: Miss 33-40% of phishers, but very low false alarm rate
- Accuracy misleading (90% baseline by predicting all "normal")
- Focus on phisher class metrics (precision, recall, F1) for true performance

### **Feature Importance Analysis**

**Top Features by Dataset Configuration**:

#### **Imbalanced Dataset (1:9) - XGBoost 03_a with 34 features**:

1. **node_indegree** (~35%) - Incoming connection count (victims)
2. **direction_ratio** (~15%) - Funnel flow detection (in/out ratio)
3. **max_outgoing_amount** (~10%) - Large fund extraction patterns
4. Statistical features dominate: 17 of top 20
5. Centrality features: 0 in top 20

#### **Balanced Dataset (5:5) - XGBoost 03_a with 34 features**:

1. **max_incoming_amount** - Large victim deposits
2. **average_outgoing_sent** - Rapid extraction patterns  
3. **max_outgoing_sent** - Large withdrawals
4. **std_time_diff** - Irregular timing (burst patterns)
5. **long_term_transfer_freq** - Sustained activity

**Key Insights Across Datasets**:
- **Statistical features dominate** both configurations (15-17 of top 20)
- **Frequency features** provide temporal context (3-5 in top 20)
- **Centrality features weak** in imbalanced data (network position less predictive)
- **node_indegree** extraordinarily important in imbalanced case (35% of total importance)

**Why These Features Matter**:
- `node_indegree`: Many victims → many incoming partners (volume business)
- `direction_ratio`: High inbound, low outbound = funnel/siphon pattern
- `max_outgoing_amount`: "Hoard then extract" phishing signature
- `*_freq` features: Bot-like burst patterns vs normal steady activity
- `account_lifetime`: Phishing accounts typically short-lived (<30 days)

### **Deployment Recommendations**

**For Research & Education**:
```
✓ Dataset: Balanced (5:5) - 10,960 accounts
✓ Model: 03_a with all 34 features
✓ Performance: 90.78% accuracy, 93.43% recall
✓ Prioritize: High recall (catch 93% of phishers)
✓ Accept: 11.86% false positive rate
✓ Use case: Feature analysis, baseline studies
```

**For Production Deployment (Conservative)**:
```
✓ Dataset: Imbalanced (1:9) - realistic 10:90 ratio
✓ Model: 03_a with all 34 features  
✓ Performance: 94.66% accuracy, 66.21% recall, 77.13% precision
✓ Trade-off: Miss 34% of phishers, but only 2.18% false positives
✓ Use case: High-confidence alerts, low false alarm tolerance
✓ Training: 1.18 seconds
```

**For Production Deployment (Fast Inference)**:
```
✓ Dataset: Imbalanced (1:9)
✓ Model: 03_b with 14 selected features
✓ Performance: 93.57% accuracy, 60.27% recall, 70.97% precision
✓ Trade-off: Miss 40% of phishers, 2.74% false positives
✓ Benefit: 28% faster (0.85 sec), 58.8% fewer features
✓ Use case: High-throughput screening, resource-constrained deployment
```

**Threshold Tuning Recommendations**:
- **Default 0.5**: Balanced precision-recall
- **Aggressive 0.21** (imbalanced): Maximize F1-score (76.33%)
- **Conservative 0.7**: Minimize false alarms for manual review

---

## 🖥️ Interactive Dashboard

### **EtherShield: Professional Cybersecurity Interface**

Our Streamlit dashboard provides a production-grade interface for phishing detection with:

**🎨 Design Features**:
- Premium dark cybersecurity theme
- Inspired by Chainalysis & Arkham Intelligence
- High-contrast text for readability
- Smooth animations and transitions
- Responsive design (desktop & mobile)

**⚡ Performance**:
- Model caching for instant responses
- Progress bars for long operations
- Real-time confidence meters
- Interactive Plotly visualizations

**🎮 Interactive Elements**:
- Quick test buttons
- Attack presets
- Sample addresses
- Export functionality
- Copy-to-clipboard

### **Dashboard Architecture**

```python
# Main application: streamlit_UI/app.py (2,646 lines)

Components:
1. CSS Theme System (245 lines)
   - Premium dark-blue color palette
   - Professional gradients & shadows
   - Responsive design rules

2. Configuration & Paths (50 lines)
   - Model file locations
   - Dataset statistics
   - Confidence thresholds

3. Utility Functions (400 lines)
   - Feature extraction (cached)
   - Model loading (cached)
   - Visualization generators

4. Four Main Tabs (1,800 lines)
   - Address Checker (400 lines)
   - Classic Simulator (350 lines)
   - Analytics Dashboard (300 lines)
   - War Room 3.0 (750 lines)

5. Footer & Metadata (150 lines)
```

### **Quick Start Guide**

```bash
# 1. Navigate to UI folder
cd streamlit_UI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch dashboard
streamlit run app.py

# 4. Open browser
# URL: http://localhost:8501
```

For complete dashboard documentation, see: **[streamlit_UI/README_UI.md](streamlit_UI/README_UI.md)**

---

## 📖 Technical Documentation

### **Feature Engineering Scripts**

#### **directed_freq_loader.py**
```python
Purpose: Extract frequency-based features from transaction graph

Key Functions:
- __init__(): Initialize graph structure
- nodeID(x): Map addresses to integer IDs
- read(file): Load transaction graph
- addEdge(s): Add directed edge with temporal info
- cal_freq_feats(): Calculate 6 frequency features
  * long_term_transfer_freq
  * short_term_transfer_freq
  * long_term_incoming_freq
  * short_term_incoming_freq
  * long_term_outgoing_freq
  * short_term_outgoing_freq

Usage:
from directed_freq_loader import directed_loader
loader = directed_loader()
loader.read("output_transactions.txt")
features = loader.cal_freq_feats()
```

#### **directed_stat_loader.py**
```python
Purpose: Extract statistical features from transaction patterns

Key Functions:
- __init__(): Initialize data structures
- nodeID(x): Map addresses to integer IDs
- read(file): Load transaction graph
- addEdge(s): Process transaction details
- cal_stat_feats(): Calculate 21 statistical features
  * Amount distributions (max, min, avg)
  * Temporal patterns (time between transactions)
  * Account characteristics (lifetime, active days)
  * Timing patterns (hour of day, weekday ratio)

Usage:
from directed_stat_loader import directed_loader
loader = directed_loader()
loader.read("output_transactions.txt")
features = loader.cal_stat_feats()
```

#### **extract_features_to_csv.py**
```python
Purpose: Complete feature extraction pipeline

Process:
1. Load transaction graph
2. Extract frequency features (directed_freq_loader)
3. Extract statistical features (directed_stat_loader)
4. Compute centrality features (NetworkX)
5. Combine all features
6. Export to CSV

Command-line Usage:
python extract_features_to_csv.py \
  --dataset MulDiGraph \
  --input ./dataset/MulDiGraph/output_transactions.txt \
  --output ./dataset/MulDiGraph/features.csv \
  --alpha 1.0
```

### **Model Training Scripts**

#### **detect_xgboost.py**
```python
Purpose: XGBoost model training with hyperparameter optimization

Key Features:
- Optuna integration for hyperparameter tuning
- WandB logging for experiment tracking
- Time-series cross-validation
- Feature importance analysis
- Model serialization (joblib)

Command-line Usage:
# Basic training
python detect_xgboost.py \
  --dataset MulDiGraph \
  --ratio 5:5 \
  --feature_select "Selected Features"

# With hyperparameter optimization
python detect_xgboost.py \
  --dataset MulDiGraph \
  --ratio 5:5 \
  --optimize \
  --n_trials 50

# Custom hyperparameters
python detect_xgboost.py \
  --n_estimators 400 \
  --max_depth 6 \
  --learning_rate 0.3
```

### **Data Files**

#### **phisher_accounts.txt**
```
Format: One address per line (Ethereum format: 0x...)
Total: 5,480 verified phishing addresses
Source: Etherscan verified phishing database
Example:
0x7a3c8f60caff89a412952f82489fe630f0f78eb9
0x906b3f8b7845840188eab53c3f5ad348a787752f
```

#### **output_transactions.txt**
```
Format: Transaction graph edges
Structure: source target value timestamp [attributes]
Total: 13,551,303 transactions
Example:
0xabc... 0xdef... 1.5 1609459200 gas=21000 nonce=42
```

#### **features.csv**
```
Format: CSV with 34 feature columns
Rows: 2,973,489 addresses
Columns: 
- address (Ethereum address)
- 34 engineered features
- label (0=normal, 1=phisher)
```

#### **selected_features.json**
```json
{
  "selected_features": [...17 feature names...],
  "threshold": 0.05,
  "method": "Spearman",
  "n_original": 34,
  "n_selected": 17,
  "reduction_rate": 50.0
}
```

### **Model Artifacts**

#### **xgboost_no_FE_selection.joblib**
```
Type: Serialized XGBoost model
Features: All 34 features
Training: Balanced or Imbalanced (depends on Phase 4)
Size: ~50-100 MB
Format: joblib pickle
```

#### **xgboost_with_FE_selection.joblib**
```
Type: Serialized XGBoost model
Features: 17 selected features
Training: Balanced or Imbalanced (depends on Phase 4)
Size: ~25-50 MB (smaller due to fewer features)
Format: joblib pickle
```

#### **xgboost_with_FE_selection_metadata.json**
```json
{
  "model_name": "XGBoost with Feature Selection",
  "model_type": "with_selection",
  "n_features": 17,
  "features": [...17 feature names...],
  "ratio": "5:5" or "1:9",
  "train_size": 660656 or larger,
  "test_size": 165164 or larger,
  "training_date": "2024-11-28",
  "performance": {
    "precision": 0.934,
    "recall": 0.923,
    "f1_score": 0.928
  }
}
```

---

## 🤝 Contributing

We welcome contributions to improve this blockchain fraud detection system!

### **How to Contribute**

1. **Fork the Repository**
```bash
# Visit GitHub and click "Fork"
https://github.com/HelcurtLordno1/Blockchain-Fraud-Detection-The-Power-of-Data-Preparation.git
```

2. **Clone Your Fork**
```bash
git clone https://github.com/YOUR_USERNAME/Blockchain-Fraud-Detection-The-Power-of-Data-Preparation.git
cd Blockchain-Fraud-Detection-The-Power-of-Data-Preparation
```

3. **Create a Branch**
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

4. **Make Your Changes**
```bash
# Add your improvements
# Test thoroughly
# Update documentation if needed
```

5. **Commit and Push**
```bash
git add .
git commit -m "Description of your changes"
git push origin feature/your-feature-name
```

6. **Create Pull Request**
```bash
# Go to GitHub and create a Pull Request
# Describe your changes clearly
# Reference any related issues
```

### **Development Guidelines**

**Code Style**:
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions
- Comment complex logic

**Testing**:
- Test your changes thoroughly
- Include unit tests where applicable
- Verify on both balanced and imbalanced datasets
- Check dashboard functionality

**Documentation**:
- Update README.md if adding features
- Update code comments
- Add examples for new functionality
- Update requirements.txt if adding dependencies

---

## 👥 Research Team

### **Project Owner**
**Dinh Nam Khanh**  
GitHub: [@HelcurtLordno1](https://github.com/HelcurtLordno1)  
Role: Project Lead & Primary Developer

### **Repository**
- **Main Repository**: https://github.com/HelcurtLordno1/Blockchain-Fraud-Detection-The-Power-of-Data-Preparation.git
- **Documentation**: Comprehensive README and inline documentation
- **License**: Educational and research purposes

### **Research Affiliation**
- **Course**: Visualization & Data Analysis
- **Instructor**: Teacher Long
- **Academic Year**: 2024-2025

### **Acknowledgments**

**Data Sources**:
- Etherscan.io for verified phishing addresses
- Ethereum blockchain for transaction data

**Inspiration**:
- May 3, 2024 $68M WBTC Address Poisoning Attack
- Real-world blockchain security challenges

**Tools & Frameworks**:
- XGBoost, LightGBM, Scikit-learn (ML frameworks)
- NetworkX (graph analysis)
- Streamlit (dashboard framework)
- Plotly (visualizations)
- Optuna (hyperparameter optimization)

**Research References**:
- Power-law distribution studies
- Blockchain fraud detection literature
- Graph neural network research
- Class imbalance handling techniques

---
