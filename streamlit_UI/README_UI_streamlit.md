# EtherShield: Ethereum Phishing Detector UI

## 🛡️ Overview

EtherShield is an interactive web-based dashboard for demonstrating machine learning-based phishing detection on Ethereum addresses. Built on thesis research addressing power-law distributions and extreme class imbalance (0.184% or 1:543) in blockchain fraud detection.

**Key Features**:
- 🔍 Real-time phishing detection with ML models
- 🎮 Interactive attack simulation with presets
- 📊 Comprehensive analytics dashboard
- ⚔️ War Room 3.0: Full adversarial testing game
- 🎨 Premium dark cybersecurity theme

---

## 🚀 Setup Guide

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or VS Code
- Models trained from notebooks (03_a and 03_b)
- Utility modules automatically included (`model_utils.py`, `data_utils.py`, `ui_components.py`)

### Step 1: Train Your Models

Before running the UI, you **must** train the machine learning models.

#### Option A: Train Model Without Feature Selection (Recommended First)
1. Open Jupyter/VS Code
2. Navigate to: `notebook_process_data/03_a_models_training_no_FE_selection.ipynb`
3. **Run ALL cells from top to bottom**
4. Look for the cell that saves the model (Part 7.7)
5. Verify model is saved to: `models/save_models/xgboost_no_FE_selection.joblib`

#### Option B: Train Model With Feature Selection
1. Open Jupyter/VS Code
2. Navigate to: `notebook_process_data/03_b_models_training_with_FE_selection.ipynb`
3. **Run ALL cells from top to bottom**
4. Look for the cell that saves the model (Part 8.5)
5. Verify model is saved to: `models/save_models/xgboost_with_FE_selection.joblib`

**Important**: Run at least ONE of these notebooks to generate a model!

### Step 2: Install Dependencies

Open PowerShell and navigate to the streamlit_UI folder:

```powershell
cd streamlit_UI
pip install -r requirements.txt
```

Wait for installation to complete (~2-3 minutes).

**✓ Verify Installation**:
```powershell
# Test that new utility modules import correctly
python -c "from model_utils import load_model_with_fallback; from data_utils import load_pickle_file; from ui_components import render_metric_card; print('✓ All imports successful!')"
```

Expected output: `✓ All imports successful!`

### Step 3: Run the Application

Still in the streamlit_UI folder, run:

```powershell
streamlit run app.py
```

The app should automatically open in your browser at: `http://localhost:8501`

If it doesn't open automatically, manually navigate to: `http://localhost:8501`

### Step 4: Explore the Interface

Once the app loads:

1. **Select Model Mode** (sidebar):
   - "No Feature Selection (All 34)" - Uses all 34 engineered features
   - "With Feature Selection (~20)" - Uses ~20 selected features

2. **Try the Address Checker**:
   - Click "Normal Example" or "Phisher Example" buttons
   - Or enter a custom Ethereum address
   - Click "Analyze Address" to see results

3. **Explore Other Tabs**:
   - 🎮 Classic Simulator - Configure and test attack patterns
   - 📊 Analytics Dashboard - View dataset statistics
   - ⚔️ War Room 3.0 - Full adversarial game experience

---

## ✅ Verification Checklist

Before running the UI, verify:

- [ ] **At least one model file exists** in `../models/save_models/`:
  - [ ] `xgboost_no_FE_selection.joblib` (from 03_a), OR
  - [ ] `xgboost_with_FE_selection.joblib` (from 03_b)

- [ ] **All dependencies installed**:
  ```powershell
  pip list | Select-String "streamlit|xgboost|plotly|networkx"
  ```
  Should show: streamlit, xgboost, plotly, networkx

- [ ] **Utility modules present** in `streamlit_UI/`:
  - [ ] `model_utils.py` - Model loading and prediction utilities
  - [ ] `data_utils.py` - Data loading and validation utilities
  - [ ] `ui_components.py` - Reusable UI components

- [ ] **You're in the correct directory**:
  ```powershell
  pwd
  ```
  Should end with: `...\Data-Prep-Project-main\streamlit_UI`

- [ ] **Imports work correctly**:
  ```powershell
  python -c "import model_utils, data_utils, ui_components; print('✓ OK')"
  ```

---

## 🐛 Common Issues & Solutions

### Issue 1: "Model file not found"
**Problem**: App shows error that model doesn't exist

**Solution**:
1. Go back to Step 1
2. Run notebook 03_a or 03_b completely
3. Check the cell output - it should show "✓ Model saved successfully!"
4. Verify file exists at: `models/save_models/xgboost_no_FE_selection.joblib`

**NOTE**: The app now uses `load_model_with_fallback()` from `model_utils.py` which provides helpful error messages telling you exactly which notebook to run.

### Issue 2: "ModuleNotFoundError: No module named 'model_utils'"
**Problem**: Cannot import utility modules

**Solution**:
```powershell
# Make sure you're in the streamlit_UI directory
cd streamlit_UI
pwd  # Should show: ...\Data-Prep-Project-main\streamlit_UI

# Verify files exist
ls model_utils.py, data_utils.py, ui_components.py

# Test imports
python -c "import model_utils, data_utils, ui_components; print('✓ OK')"
```

### Issue 3: "ModuleNotFoundError" (Other packages)
**Problem**: Missing Python packages

**Solution**:
```powershell
cd streamlit_UI
pip install -r requirements.txt --force-reinstall
```

### Issue 4: Port Already in Use
**Problem**: Error says port 8501 is busy

**Solution**:
```powershell
# Use a different port
streamlit run app.py --server.port 8502
```
Then open: `http://localhost:8502`

### Issue 5: App Loads But Shows Errors
**Problem**: UI loads but functionality doesn't work

**Solution**:
1. Check terminal for error messages (now with structured logging!)
2. Clear Streamlit cache:
   ```powershell
   streamlit cache clear
   ```
3. Restart the app (Ctrl+C, then `streamlit run app.py`)

### Issue 6: Import Errors After Update
**Problem**: App worked before but now shows import errors

**Solution**:
```powershell
# The app now uses new utility modules
# Make sure they exist in streamlit_UI folder:
ls model_utils.py  # Should exist
ls data_utils.py   # Should exist
ls ui_components.py # Should exist

# If missing, you may need to pull latest changes:
git pull origin notebook
```

### Issue 7: Dark Text on Dark Background
**Problem**: Some text is hard to read

**Solution**: This has been fixed in the latest version. If you still see issues:
1. Pull latest changes from repository
2. Restart the Streamlit app
3. All text now uses light colors (#f1f5f9, #cbd5e1, #94a3b8) for proper contrast

---

## 📁 Expected File Structure

After setup, you should have:

```
Data-Prep-Project-main/
├── models/
│   └── save_models/
│       ├── xgboost_no_FE_selection.joblib          ← Required
│       ├── xgboost_with_FE_selection.joblib        ← Required
│       └── xgboost_with_FE_selection_metadata.json ← Required
├── features_engineering/
│   ├── directed_freq_loader.py    
│   └── directed_stat_loader.py    
├── notebook_process_data/
│   ├── 03_a_models_training_no_FE_selection.ipynb
│   └── 03_b_models_training_with_FE_selection.ipynb
└── streamlit_UI/
    ├── app.py                  # Main Streamlit application (updated)
    ├── model_utils.py          # Model utilities (242 lines)
    ├── data_utils.py           # Data utilities (326 lines)
    ├── ui_components.py        # UI components (420 lines)
    ├── requirements.txt        # Python dependencies
    ├── README_UI_streamlit.md  # README understand to run
    ├── data/                   # (Created automatically)
    └── saved_outputs/          # (Created automatically - exported results)
```

### 🆕 New Utility Modules

**model_utils.py** (242 lines):
- `load_model_with_fallback()` - Defensive model loading
- `load_model_metadata()` - JSON metadata loading
- `predict_with_probability()` - Predictions with confidence
- `get_feature_importance()` - Extract top features
- `validate_model_compatibility()` - Dimension checking

**data_utils.py** (326 lines):
- `load_pickle_file()`, `load_json_file()`, `load_features_csv()`
- `extract_features_for_address()` - Feature extraction
- `validate_feature_vector()` - NaN/Inf checking
- `check_data_pipeline_status()` - Pipeline validation
- `generate_sample_features()` - Test data generation

**ui_components.py** (420 lines):
- 12 reusable UI rendering functions
- `render_metric_card()`, `render_status_badge()`
- `render_probability_gauge()`, `render_confusion_matrix()`
- `render_feature_importance_chart()`, `render_section_header()`
- Consistent styling and type safety

---

## ✨ UI Features & Functionality

### 🎨 Professional Dark Cybersecurity Theme
- **Premium color scheme**: Dark blue backgrounds with cyan/emerald accents
- **High contrast text**: All text optimized for readability on dark backgrounds
- **Smooth animations**: Hover effects, glow effects, and transitions
- **Responsive design**: Works on desktop and mobile devices

### 📑 Four Main Tabs

#### 🔍 Tab 1: Address Checker

**Quick Test Buttons**:
- 📗 **Normal Example** - Generates fresh legitimate address patterns each click
- 📕 **Phisher Example** - Generates fresh phishing address patterns each click
- 🎲 **Random Address** - Generates random Ethereum address for testing
- 📋 **View Examples** - Shows 9 pre-configured addresses (4 phishing + 5 legitimate)

**Analysis Features**:
- Real-time progress bar during analysis
- Confidence scoring with visual indicators (color-coded risk levels)
- Feature importance analysis with top 10 features
- Interactive radar chart for feature visualization
- Network graph showing transaction relationships
- Detailed risk assessment with recommendations
- One-click address copy button

**Risk Levels**:
- 🔴 **CRITICAL** - Confidence >90%
- 🟠 **HIGH** - Confidence 70-90%
- 🟡 **MEDIUM** - Confidence 30-70%
- 🟢 **LOW** - Confidence <30%

#### 🎮 Tab 2: Classic Simulator

**Attack Configuration**:
- Spam transactions slider (0-100)
- Vanity similarity slider (6-12 characters)
- Funnel inbound ratio (0.0-1.0)
- Burst pattern toggle
- Funnel flow toggle

**Attack Presets** (Quick load buttons):
1. **🟢 Low Risk Attack**
   - 5 spam transactions
   - 6 character similarity
   - 30% funnel ratio
   
2. **🟡 Medium Risk Attack**
   - 20 spam transactions
   - 8 character similarity
   - 60% funnel ratio
   
3. **🔴 High Risk Attack** (May 3, 2024 inspired)
   - 50 spam transactions
   - 12 character similarity
   - 90% funnel ratio

**Results Display**:
- Detection probability gauge
- Attack pattern analysis
- Feature comparison charts
- Real-world attack context

#### 📊 Tab 3: Analytics Dashboard

**Dataset Overview**:
- Total addresses: 2,973,489
- Total transactions: 13,551,303
- Verified phishers: 5,480
- Imbalance ratio: 1:543

**Visualizations**:
- Power-law degree distribution (log scale)
- Heavy-tailed transaction values
- Model performance comparison (03_a vs 03_b)
- Feature count comparison

**Feature Categories**:
- 📊 Frequency Features (6) - Temporal burstiness detection
- 📈 Statistical Features (21) - Account behavior patterns
- 🌐 Centrality Features (7) - Graph structure analysis

**Export Options**:
- 📊 Model metrics (JSON format)
- 📈 Dashboard report (TXT format)

**Sample Data Viewer**:
- Example feature comparison table
- Shows differences between phisher and normal addresses
- Highlights key detection indicators

#### ⚔️ Tab 4: War Room 3.0 - Breach Protocol

**Two Game Modes**:

**🔴 Attacker Mode (Red Team)**:
- Design multi-day phishing campaigns (1-30 days)
- Choose from 9 attack types:
  - Address Poisoning
  - Dust Spam
  - Fake Airdrop
  - ICE Phishing
  - Flash Loan Attack
  - Sybil Network
  - Mixer Obfuscation
  - Gas Price Manipulation
  - Custom Attack
- Configure attack parameters:
  - Transaction intensity (1-50 tx/day)
  - Burst patterns (multiplier: 1-10x)
  - Funnel ratio (0.0-1.0)
  - Dust transactions (0-100)
  - Stealth mode (detection evasion after day 7)
- Day-by-day simulation with risk tracking
- Success threshold: Keep phishing risk <50% for campaign duration
- Hall of Fame leaderboard for successful bypasses

**🛡️ Defender Mode (Blue Team)**:
- Build layered security defenses:
  - 🔐 Hardware Wallet (Ledger/Trezor)
  - ⚡ Transaction Simulation (WalletGuard)
  - 📋 Address Whitelisting
  - 🔄 Regular Approval Audits (Revoke.cash)
  - 🔒 Two-Factor Authentication
  - ✅ Verified Contracts Only
  - 🐌 Slow & Steady Behavior
- Test defenses against 6 real-world attacks:
  - Address Poisoning
  - Dust Attack
  - Fake Airdrop
  - ICE Phishing
  - Flash Loan
  - Normal Transaction (control)
- Defense effectiveness scoring
- Badge/achievement system
- Battle timeline visualization

**War Room Features**:
- Real-time attack/defense simulation
- Battle statistics and charts
- Export battle reports
- Persistent leaderboard across sessions
- Day-by-day progression tracking

### 🎯 User Experience Enhancements

**Performance Optimizations**:
- Feature extraction caching for faster repeated analysis
- Model loading cached with `@st.cache_resource`
- Instant response for previously analyzed addresses

**Interactive Elements**:
- Progress bars for all long-running operations
- Real-time confidence meters with delta animations
- Smooth hover effects on cards and buttons
- Color-coded status indicators throughout
- Copyable code blocks for addresses

**Session State Management**:
- Persistent leaderboard data
- Battle log history
- Defender badges tracking
- Campaign results storage
- UI panel visibility toggles

---

## 🎯 Understanding Model Modes

### No Feature Selection (03_a)
- **Features**: All 34 features (frequency + statistical + centrality)
- **Training**: Uses dataset from get_dataset_5_5.ipynb (50:50) OR get_dataset_1_9.ipynb (10:90)
- **Performance**: Maximum information available to model
- **Use case**: Baseline performance, comprehensive feature analysis
- **Model file**: `xgboost_no_FE_selection.joblib`

### With Feature Selection (03_b)
- **Features**: ~20 selected features (based on Spearman correlation)
- **Training**: Uses SAME dataset as 03_a (5_5 or 1_9)
- **Performance**: Similar accuracy with 41% fewer features
- **Use case**: Production deployment, faster inference
- **Model file**: `xgboost_with_FE_selection.joblib`

**Important Note**: The class balance (50:50 or 10:90) is determined by which get_dataset notebook you ran BEFORE training, not by which model (03_a vs 03_b) you use. Both models trained on the same dataset will have the same balance.

## 🔬 Technical Details

### Code Architecture (Updated)

**Modular Design**:
- **app.py** (2,672 lines): Main application with improved structure
  - Imports from `model_utils`, `data_utils`, `ui_components`
  - Uses defensive loading functions
  - Better error handling with helpful messages
  
- **model_utils.py** (242 lines): Model operations
  - Type hints: `load_model_with_fallback(model_path: str) -> Optional[Any]`
  - Structured logging with `logging.getLogger(__name__)`
  - Comprehensive docstrings with examples
  
- **data_utils.py** (326 lines): Data operations
  - Type hints: `load_pickle_file(file_path: str) -> Optional[Any]`
  - File validation before loading
  - Error messages reference which notebook to run
  
- **ui_components.py** (420 lines): UI components
  - 12 reusable render functions
  - Consistent styling across dashboard
  - Type-safe component interfaces

### Feature Categories (34 Total)

1. **Frequency Features (6)**:
   - Long/short-term transfer frequencies
   - Captures temporal burstiness
   - Detects bot-like behavior
   - **Extracted by**: `directed_freq_loader.py` (now with full documentation)

2. **Statistical Features (21)**:
   - Degree, balance, lifetime
   - Transaction amounts & timing
   - Detects funnel flow patterns
   - **Extracted by**: `directed_stat_loader.py` (now with full documentation)

3. **Centrality Features (7)**:
   - Katz, degree, closeness centrality
   - Clustering coefficient
   - Detects isolated victim structures
   - **Calculated by**: NetworkX graph algorithms

### Model Architecture
- **Algorithm**: XGBoost (Gradient Boosting)
- **Trees**: 400 boosting rounds
- **Max Depth**: 6
- **Learning Rate**: 0.3
- **Random State**: 42 (reproducible results)
- **Serialization**: joblib (efficient for sklearn-compatible models)
- **Loading**: Via `load_model_with_fallback()` with defensive checks

## 📊 Dataset Statistics

- **Total Addresses**: 2,973,489
- **Total Transactions**: 13,551,303
- **Verified Phishers**: 5,480
- **Phisher Rate**: 0.184% (1:543 imbalance)
- **Inactive Addresses**: 89.4% (≤2 transactions)
- **Super-nodes**: 27 addresses with >100K transactions

## 💡 Pro Tips & Best Practices

### For First-Time Users
1. **Start with Address Checker**: Click "Normal Example" and "Phisher Example" to see how the model works
2. **Explore Sample Addresses**: Use "View Examples" to test pre-configured addresses
3. **Check Logs**: The terminal now shows structured logging with helpful messages
4. **Understand Errors**: Error messages now tell you exactly which notebook to run if files are missing

### For Developers
1. **Modular Code**: Import utilities as needed:
   ```python
   from model_utils import load_model_with_fallback
   from data_utils import load_pickle_file
   from ui_components import render_metric_card
   ```
2. **Type Safety**: All functions have type hints - use IDE autocomplete!
3. **Documentation**: Check docstrings with `help(function_name)`
4. **Logging**: Use `logger.info()`, `logger.error()` for debugging
5. **Testing**: Run validation tests from `QUICK_CHECKLIST.md`

### Code Quality Features
✅ **Type Hints**: All functions annotated with types
✅ **Docstrings**: Every function has Args/Returns/Example
✅ **Logging**: Structured logging throughout
✅ **Defensive Checks**: File existence validation before loading
✅ **Helpful Errors**: Messages guide you to correct notebook
✅ **Reproducibility**: Random seed set to 42 everywhere
3. **Try the Simulator**: Load attack presets to understand different threat levels
4. **Check Analytics**: View dataset statistics and model performance metrics
5. **Play War Room**: Test both Attacker and Defender modes to understand adversarial scenarios

### For Developers
- **Feature extraction is cached**: Same address returns instant results
- **Session state persists**: Leaderboard and battle logs saved during session
- **Models are cached**: First load takes ~5-10 seconds, subsequent loads are instant
- **Export functionality**: Save results as JSON or TXT for further analysis

### For Security Researchers
- **War Room provides insights**: Test detection robustness with multi-day campaigns
- **Stealth mode simulation**: Understand evasion techniques after day 7
- **Defense stacking**: See how multiple security layers compound protection
- **Real attack patterns**: Presets based on actual attacks like May 3, 2024 incident

### UI Navigation Tips
- **Sidebar controls**: Switch between models anytime
- **Tab persistence**: State maintained when switching tabs
- **Progress bars**: Show processing status for long operations
- **Copy buttons**: Quick copy for addresses and results
- **Export buttons**: Download metrics and reports

---

## 🛠️ Advanced Customization

### Connecting to Real Etherscan API

For production use with real blockchain data:

1. **Install additional dependencies**:
```powershell
pip install etherscan-python web3
```

2. **Get Etherscan API key** from https://etherscan.io/apis

3. **Modify feature extraction** in `app.py`:
```python
from etherscan import Etherscan
eth = Etherscan(YOUR_API_KEY)

def extract_features_from_address(address):
    # Fetch real transaction data
    transactions = eth.get_normal_txs_by_address(
        address, 
        startblock=0, 
        endblock=99999999, 
        sort='asc'
    )
    # Calculate features from real data
    # ... your feature engineering logic
```

### Adding Custom Attack Types

To add new attack scenarios in War Room:

1. Edit the attack type list in `simulate_war_room_scenario()`
2. Define feature modifications for your attack pattern
3. Add attack description and transaction details
4. Update the attack selector UI

### Theming & Styling

All colors are defined in CSS variables at the top of `app.py`:
- `--bg-primary`: Main background color
- `--accent-cyan`: Primary accent color
- `--success`: Success state color (emerald green)
- `--danger`: Danger state color (crimson red)

Modify these variables to customize the entire theme.

## ⚠️ Important Notes

### Demo Mode vs Production
- **Current mode**: Demo with simulated features for visualization
- **No real blockchain queries**: Without API integration, addresses are not queried from Etherscan
- **Consistent results**: Same address generates same features (seeded random generation)
- **Privacy**: All computations are local, no data sent externally

### Model File Requirements
The app expects models at:
- `../models/save_models/xgboost_no_FE_selection.joblib`
- `../models/save_models/xgboost_with_FE_selection.joblib`
- `../models/save_models/xgboost_with_FE_selection_metadata.json`

If models are not found, the app will display setup instructions with links to training notebooks.

### Performance Characteristics
- **First load**: 5-10 seconds (model loading)
- **Cached operations**: Instant response for repeated addresses
- **War Room simulations**: 2-5 seconds per campaign
- **Chart rendering**: Real-time with Plotly

### Browser Compatibility
- **Best experience**: Chrome, Edge, Firefox (latest versions)
- **Mobile responsive**: Works on tablets and phones
- **JavaScript required**: For interactive charts and animations

---

## 🆘 Additional Troubleshooting

### Slow War Room Simulations
**Problem**: Multi-day campaigns take too long

**Solution**:
- Reduce campaign duration (try 1-7 days first)
- Lower transaction intensity
- Close other browser tabs

### Charts Not Displaying
**Problem**: Plotly visualizations don't appear

**Solution**:
1. Check browser console for JavaScript errors
2. Disable browser extensions (ad blockers)
3. Clear browser cache
4. Ensure `plotly` is installed: `pip install plotly`

### Cached Data Issues
**Problem**: Old results showing after model update

**Solution**:
```powershell
streamlit cache clear
# Restart the app
streamlit run app.py
```

### Memory Issues
**Problem**: App becomes slow over time

**Solution**:
1. Restart the Streamlit app (Ctrl+C, then rerun)
2. Clear browser cache
3. Reduce number of stored leaderboard entries

## 🔄 Updates & Maintenance

### Updating the Application

1. **Pull latest changes** from repository
2. **Update dependencies**:
   ```powershell
   pip install -r requirements.txt --upgrade
   ```
3. **Clear Streamlit cache**:
   ```powershell
   streamlit cache clear
   ```
4. **Restart the application**:
   ```powershell
   streamlit run app.py
   ```

### When to Retrain Models

Retrain models if:
- Feature engineering logic changes in notebooks
- New features are added to the 34-feature set
- Dataset balance changes (switching between 5_5 and 1_9)
- XGBoost hyperparameters are tuned

### Backup Recommendations

Regularly backup:
- `models/save_models/*.joblib` - Trained models
- `streamlit_UI/saved_outputs/` - Exported results
- War Room leaderboard data (stored in session state)

---

## 🎓 Educational Use Cases

### For Students
- **Learn ML concepts**: See how features affect predictions
- **Understand imbalance**: Compare 50:50 vs 10:90 dataset performance
- **Explore attacks**: War Room teaches real phishing techniques
- **Feature importance**: Radar charts show which features matter most

### For Instructors
- **Live demonstrations**: Show model predictions in real-time
- **Attack simulations**: Teach security concepts with interactive examples
- **Performance metrics**: Compare different model architectures
- **Export capabilities**: Generate reports for assignments

### For Researchers
- **Adversarial testing**: War Room provides systematic attack evaluation
- **Feature analysis**: Identify which features are most discriminative
- **Power-law insights**: Visualize distribution challenges in blockchain data
- **Baseline comparisons**: Test new features against existing 34-feature set

---

## 🎨 Design Philosophy

### Visual Theme
- **Inspired by**: Chainalysis, Arkham Intelligence, Black Hat presentations
- **Color palette**: Professional dark blues with cyan/emerald accents
- **Typography**: Clean, high-contrast text for readability
- **Animations**: Subtle hover effects and smooth transitions
- **Accessibility**: All text optimized for visibility on dark backgrounds

### User Experience Principles
1. **Immediate feedback**: Progress bars for all operations
2. **Clear hierarchy**: Important information prominently displayed
3. **Consistent patterns**: Similar interactions work the same way across tabs
4. **Helpful guidance**: Expanders and tooltips explain features
5. **Error prevention**: Validation and clear requirements

---

## 📚 Technical References

### Dataset & Research
- **Data Source**: Etherscan Verified Phishing Addresses (5,480 confirmed phishers)
- **Total Scale**: 2.97M addresses, 13.55M transactions
- **Research Focus**: Power-law distributions and extreme imbalance (1:543)
- **Inspired by**: May 3, 2024 $68M WBTC Address Poisoning Attack

### Technology Stack
- **Frontend**: Streamlit 1.31.0 (Python web framework)
- **ML Model**: XGBoost (Gradient Boosting)
- **Visualization**: Plotly (interactive charts), NetworkX (graph analysis)
- **Data Processing**: Pandas, NumPy
- **Styling**: Custom CSS with CSS variables

### Key Innovations
1. **Power-Law Aware**: Uses Spearman correlation (rank-based) for feature selection
2. **Temporal Dynamics**: Captures burstiness via frequency features
3. **Graph Structure**: Leverages centrality metrics for network analysis
4. **Real Dataset**: Not simulated - uses actual Etherscan verified addresses

---

## 📧 Support & Contribution

### Getting Help
1. **Check this README**: Most common issues covered above
2. **Review terminal logs**: Error messages provide specific details
3. **Verify setup**: Follow checklist to ensure proper configuration
4. **Test notebooks**: Ensure models trained successfully

### Reporting Issues
When reporting problems, include:
- Python version (`python --version`)
- Installed packages (`pip list`)
- Error messages from terminal
- Steps to reproduce the issue

---

## 📄 License & Disclaimer

**Purpose**: Educational and research tool for understanding blockchain phishing detection

**Disclaimer**:
- ⚠️ Not financial advice
- ⚠️ Not for production deployment without proper testing
- ⚠️ Demo uses simulated features unless API integration added
- ⚠️ Model performance depends on training data quality

**Academic Use**: Free for educational and research purposes

---

## 🙏 Acknowledgments

This project addresses real-world challenges in Ethereum phishing detection:
- **Extreme class imbalance** (0.184% phishers, ratio 1:543)
- **Power-law distributions** (89.4% addresses have ≤2 transactions)
- **Heavy-tailed values** (mean distorted by outliers, kurtosis >250,000)
- **Temporal burstiness** (bot-like attack patterns)
- **Funnel flows** (money collection patterns)

Built with research-grade rigor and production-ready polish.

---

## 🚀 Quick Reference

### Startup Command
```powershell
cd streamlit_UI
streamlit run app.py
```

### Essential Files
- `app.py` - Main application (2,646 lines)
- `requirements.txt` - Dependencies
- `../models/save_models/*.joblib` - Trained models

### Key Shortcuts
- **Ctrl+C** - Stop Streamlit server
- **R** (in browser) - Rerun the app
- **C** (in browser) - Clear cache

### Default URLs
- **Local**: http://localhost:8501
- **Alternative port**: http://localhost:8502

---

**Version**: 2.0 - War Room Edition  
**Last Updated**: November 27, 2024  
**Status**: Research Demo with Full Adversarial Testing  
**Theme**: Premium Dark Cybersecurity Aesthetic

🛡️ **Stay vigilant. The blockchain never sleeps.**
