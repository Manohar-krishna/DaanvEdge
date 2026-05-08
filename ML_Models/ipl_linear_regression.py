"""
IPL Dataset - Linear Regression Model
Full Pipeline: Preprocessing → Feature Engineering → Processing →
               Training/Testing/Validation → Evaluation → Optimization → Prediction
Packages: pandas, numpy, scikit-learn, pytorch
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)


# ─────────────────────────────────────────────────────────────
# MOCK IPL DATASET (replace with: df = pd.read_csv("Ipl_dataset.csv"))
# ─────────────────────────────────────────────────────────────
def load_ipl_dataset():
    """
    Simulates an IPL match dataset.
    Columns: team, opponent, venue, balls_faced, wickets_in_hand,
             current_run_rate, batting_avg, bowling_avg → target: runs_scored
    """
    n = 500
    df = pd.DataFrame({
        "team":              np.random.choice(["MI", "CSK", "RCB", "KKR", "SRH", "DC"], n),
        "opponent":          np.random.choice(["MI", "CSK", "RCB", "KKR", "SRH", "DC"], n),
        "venue":             np.random.choice(["Wankhede", "Chepauk", "Eden", "Chinnaswamy"], n),
        "balls_faced":       np.random.randint(60, 120, n),
        "wickets_in_hand":   np.random.randint(1, 10, n),
        "current_run_rate":  np.round(np.random.uniform(5.0, 12.0, n), 2),
        "batting_avg":       np.round(np.random.uniform(20.0, 60.0, n), 2),
        "bowling_avg":       np.round(np.random.uniform(18.0, 40.0, n), 2),
        "runs_scored":       np.random.randint(80, 220, n)          # TARGET
    })
    # Inject realistic correlation: more balls + higher RR → more runs
    df["runs_scored"] += (df["balls_faced"] * df["current_run_rate"] * 0.1).astype(int)
    df["runs_scored"] = df["runs_scored"].clip(80, 250)
    return df


# ─────────────────────────────────────────────────────────────
# STEP 1 — DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────
def step1_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Goal : Produce a clean, numeric-only DataFrame.
    Actions:
      - Inspect shape, dtypes, null counts
      - Drop duplicates
      - Encode categorical columns with LabelEncoder
      - Impute missing values (median for numeric, mode for categorical)
    """
    print("\n" + "="*60)
    print("STEP 1 — DATA PREPROCESSING")
    print("="*60)
    print(f"Raw shape     : {df.shape}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"Null counts:\n{df.isnull().sum()}")

    # Drop exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Impute missing values
    for col in df.select_dtypes(include="number").columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode categoricals
    le = LabelEncoder()
    for col in ["team", "opponent", "venue"]:
        df[col] = le.fit_transform(df[col])

    print(f"\nClean shape   : {df.shape}")
    print("Preprocessing complete ✓")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def step2_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Goal : Create domain-relevant synthetic features.
    New features:
      - projected_runs  : balls_faced × current_run_rate / 6
      - batting_pressure: batting_avg / (wickets_in_hand + 1)
      - run_rate_delta  : current_run_rate - bowling_avg / 4
    """
    print("\n" + "="*60)
    print("STEP 2 — FEATURE ENGINEERING")
    print("="*60)

    df["projected_runs"]   = (df["balls_faced"] * df["current_run_rate"] / 6).round(2)
    df["batting_pressure"] = (df["batting_avg"] / (df["wickets_in_hand"] + 1)).round(2)
    df["run_rate_delta"]   = (df["current_run_rate"] - df["bowling_avg"] / 4).round(2)

    print("New features added: projected_runs, batting_pressure, run_rate_delta")
    print(f"Shape after engineering: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 3 — DATA PROCESSING
# ─────────────────────────────────────────────────────────────
def step3_data_processing(df: pd.DataFrame):
    """
    Goal : Split into X / y, scale features, prepare tensors.
    Returns sklearn arrays AND PyTorch tensors.
    """
    print("\n" + "="*60)
    print("STEP 3 — DATA PROCESSING")
    print("="*60)

    TARGET  = "runs_scored"
    FEATURES = [c for c in df.columns if c != TARGET]

    X = df[FEATURES].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)

    # StandardScaler: zero mean, unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Features used : {FEATURES}")
    print(f"X shape       : {X_scaled.shape}")
    print(f"y shape       : {y.shape}")
    print(f"y stats       → mean={y.mean():.1f}, std={y.std():.1f}, "
          f"min={y.min():.0f}, max={y.max():.0f}")

    # PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y,        dtype=torch.float32).unsqueeze(1)

    return X_scaled, y, X_tensor, y_tensor, scaler, FEATURES


# ─────────────────────────────────────────────────────────────
# STEP 4 — TRAINING, TESTING & VALIDATION
# ─────────────────────────────────────────────────────────────

# --- 4a. Scikit-learn Linear Regression ---
def step4_sklearn_train(X_scaled, y):
    """
    80/20 train-test split + 5-fold cross-validation.
    """
    print("\n" + "="*60)
    print("STEP 4a — SKLEARN LINEAR REGRESSION")
    print("="*60)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Cross-validation (R²)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="r2")
    print(f"5-Fold CV R²  : {cv_scores.round(4)} → mean={cv_scores.mean():.4f}")

    return model, X_train, X_test, y_train, y_test


# --- 4b. PyTorch Linear Regression ---
class TorchLinearRegression(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def step4_pytorch_train(X_tensor, y_tensor, input_dim: int,
                        epochs: int = 200, lr: float = 0.01):
    """
    PyTorch training loop with MSE loss and Adam optimizer.
    """
    print("\n" + "="*60)
    print("STEP 4b — PYTORCH LINEAR REGRESSION")
    print("="*60)

    # 80/20 split
    split = int(0.8 * len(X_tensor))
    X_tr, X_te = X_tensor[:split], X_tensor[split:]
    y_tr, y_te = y_tensor[:split], y_tensor[split:]

    dataset    = TensorDataset(X_tr, y_tr)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    torch_model = TorchLinearRegression(input_dim)
    criterion   = nn.MSELoss()
    optimizer   = optim.Adam(torch_model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        torch_model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            loss = criterion(torch_model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:>3}/{epochs}  Loss={epoch_loss/len(dataloader):.4f}")

    return torch_model, X_te, y_te


# ─────────────────────────────────────────────────────────────
# STEP 5 — EVALUATION
# ─────────────────────────────────────────────────────────────
def step5_evaluate(model, X_test, y_test, label="Sklearn"):
    """
    Computes MAE, MSE, RMSE, R².
    Works for both sklearn and PyTorch models.
    """
    print("\n" + "="*60)
    print(f"STEP 5 — EVALUATION [{label}]")
    print("="*60)

    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            preds = model(X_test).squeeze().numpy()
        y_true = y_test.squeeze().numpy()
    else:
        preds  = model.predict(X_test)
        y_true = y_test

    mae  = mean_absolute_error(y_true, preds)
    mse  = mean_squared_error(y_true, preds)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, preds)

    print(f"  MAE  : {mae:.4f}")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


# ─────────────────────────────────────────────────────────────
# STEP 6 — OPTIMIZATION
# ─────────────────────────────────────────────────────────────
def step6_optimization(X_scaled, y):
    """
    Compare Ridge (L2) and Lasso (L1) regularization vs plain OLS.
    Polynomial feature expansion (degree=2) is also tested.
    """
    print("\n" + "="*60)
    print("STEP 6 — OPTIMIZATION (Regularization & Polynomial Features)")
    print("="*60)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    results = {}
    experiments = {
        "OLS (baseline)" : LinearRegression(),
        "Ridge α=1.0"    : Ridge(alpha=1.0),
        "Ridge α=10.0"   : Ridge(alpha=10.0),
        "Lasso α=0.1"    : Lasso(alpha=0.1),
        "Lasso α=1.0"    : Lasso(alpha=1.0),
    }

    for name, m in experiments.items():
        m.fit(X_train, y_train)
        r2 = r2_score(y_test, m.predict(X_test))
        rmse = np.sqrt(mean_squared_error(y_test, m.predict(X_test)))
        results[name] = {"R2": round(r2, 4), "RMSE": round(rmse, 4)}
        print(f"  {name:<20} R²={r2:.4f}  RMSE={rmse:.4f}")

    # Polynomial (degree=2) with Ridge
    poly_pipeline = Pipeline([
        ("poly",  PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge", Ridge(alpha=1.0))
    ])
    poly_pipeline.fit(X_train, y_train)
    r2   = r2_score(y_test, poly_pipeline.predict(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, poly_pipeline.predict(X_test)))
    results["Poly(2)+Ridge"] = {"R2": round(r2, 4), "RMSE": round(rmse, 4)}
    print(f"  {'Poly(2)+Ridge':<20} R²={r2:.4f}  RMSE={rmse:.4f}")

    best = max(results, key=lambda k: results[k]["R2"])
    print(f"\n  ✓ Best model: {best} with R²={results[best]['R2']}")
    return results


# ─────────────────────────────────────────────────────────────
# STEP 7 — PREDICTION
# ─────────────────────────────────────────────────────────────
def step7_predict(sklearn_model, scaler, feature_names: list):
    """
    Predict runs for a single hypothetical IPL innings.
    Input is aligned to the same features used in training.
    """
    print("\n" + "="*60)
    print("STEP 7 — PREDICTION (New Match Scenario)")
    print("="*60)

    # Example: MI vs CSK at Wankhede, strong batting conditions
    # Encoded values must match LabelEncoder used in Step 1
    new_match = {
        "team"             : 1,      # MI encoded
        "opponent"         : 0,      # CSK encoded
        "venue"            : 3,      # Wankhede encoded
        "balls_faced"      : 100,
        "wickets_in_hand"  : 6,
        "current_run_rate" : 9.5,
        "batting_avg"      : 48.0,
        "bowling_avg"      : 22.0,
        "projected_runs"   : 100 * 9.5 / 6,
        "batting_pressure" : 48.0 / (6 + 1),
        "run_rate_delta"   : 9.5 - 22.0 / 4,
    }

    # Align to training feature order
    X_new = np.array([[new_match[f] for f in feature_names]], dtype=np.float32)
    X_new_scaled = scaler.transform(X_new)

    prediction = sklearn_model.predict(X_new_scaled)[0]
    print(f"  Input scenario : {new_match}")
    print(f"\n  ➜ Predicted Runs Scored : {prediction:.1f}")
    return prediction


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load
    df_raw = load_ipl_dataset()
    # df_raw = pd.read_csv("Ipl_dataset.csv")   ← swap this for real data

    # Step 1: Preprocessing
    df_clean = step1_preprocessing(df_raw)

    # Step 2: Feature Engineering
    df_feat = step2_feature_engineering(df_clean)

    # Step 3: Processing (scale, split, tensorize)
    X_scaled, y, X_tensor, y_tensor, scaler, features = step3_data_processing(df_feat)

    # Step 4a: Sklearn train
    sk_model, X_tr, X_te, y_tr, y_te = step4_sklearn_train(X_scaled, y)

    # Step 4b: PyTorch train
    torch_model, X_te_t, y_te_t = step4_pytorch_train(
        X_tensor, y_tensor, input_dim=len(features)
    )

    # Step 5: Evaluate both
    sk_metrics = step5_evaluate(sk_model, X_te, y_te, label="Sklearn LinearRegression")
    pt_metrics = step5_evaluate(torch_model, X_te_t, y_te_t, label="PyTorch LinearRegression")

    # Step 6: Optimization
    opt_results = step6_optimization(X_scaled, y)

    # Step 7: Prediction
    pred = step7_predict(sk_model, scaler, features)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE ✓")
    print("="*60)
