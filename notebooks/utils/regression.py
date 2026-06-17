import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor, SGDRegressor, QuantileRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, KFold

RANDOM_STATE = 42


class Regression:
    def __init__(self, loss_type: str = "squared"):
        """
        Initialize the Regression model trainer.

        Parameters
        ----------
        loss_type : {"squared", "huber", "mae", "log_cosh"}, default "squared"
            The family of loss functions to use where possible.
            - "squared"   : original models (MSE loss)
            - "huber"     : Huber loss (robust to outliers)
            - "mae"       : absolute error / median regression
            - "log_cosh"  : approximate log-cosh via Huber on scaled target
        """
        self.loss_type = loss_type
        self.TARGET_COL = "App A Co-Scheduled MPI Time with App B"
        self.PRESERVE_COLS = ["App A", "App B"]

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        selected_features: list,
        loss_type: str = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Train regression models and evaluate performance using LOO and
        Repeated 5-Fold cross-validation.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with features and target column.
        selected_features : list
            List of feature column names to use for training.
        loss_type : {"squared", "huber", "mae", "log_cosh"}, optional
            Override the loss type set during initialization.

        Returns
        -------
        results_df : pd.DataFrame
            DataFrame with performance metrics for each model.
        predictions_df : pd.DataFrame
            DataFrame with true values and LOO predictions per model.
        """
        if loss_type is None:
            loss_type = self.loss_type

        TARGET_COL = self.TARGET_COL
        PRESERVE_COLS = self.PRESERVE_COLS

        # ── 1. Filter to selected features + target; drop preserve cols if present ─
        cols_to_use = [c for c in selected_features if c not in PRESERVE_COLS]
        excluded = [c for c in selected_features if c in PRESERVE_COLS]

        missing = [c for c in cols_to_use if c not in df.columns]
        if missing:
            raise ValueError(f"These requested features are not in the dataframe: {missing}")
        if TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe.")

        df_filtered = df[cols_to_use + [TARGET_COL]].dropna().copy()

        print(f"\n{'='*60}")
        print("REGRESSION TRAINING & EVALUATION")
        print(f"{'='*60}")
        print(f"  Features used      : {len(cols_to_use)}  → {cols_to_use}")
        print(f"  Columns excluded   : {excluded if excluded else 'none'}")
        print(f"  Samples after dropna: {len(df_filtered)}")
        print(f"  Target             : {TARGET_COL}")

        # ── 2. Standardize features (and possibly target) ──────────────────────────
        X_raw = df_filtered[cols_to_use].values
        y = df_filtered[TARGET_COL].values.astype(float)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        # Target scaling for log_cosh (HuberRegressor needs well-conditioned target)
        if loss_type == "log_cosh":
            y_scaler = StandardScaler()
            y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
            print("  (target scaled for log_cosh)")
        else:
            y_scaler = None
            y_scaled = y  # use original

        # ── 3. Define models adapted to loss_type ───────────────────────────────────
        n_samples = len(y)
        k_neighbors = min(3, n_samples - 1)

        # Default (squared error)
        models_default = {
            "Ridge": Ridge(alpha=1.0),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
            "SVR (RBF)": SVR(kernel="rbf", C=10, epsilon=0.1),
            f"KNN (k={k_neighbors})": KNeighborsRegressor(n_neighbors=k_neighbors),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=50,
                max_depth=2,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
            ),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, max_depth=3, random_state=RANDOM_STATE, n_jobs=-1
            ),
        }

        if loss_type == "squared":
            models = models_default

        elif loss_type == "huber":
            delta = 1.35  # standard Huber constant
            models = {
                "Ridge": HuberRegressor(epsilon=delta, alpha=1.0, max_iter=1000),
                "ElasticNet": SGDRegressor(
                    loss="huber",
                    penalty="elasticnet",
                    alpha=0.1,
                    l1_ratio=0.5,
                    max_iter=10000,
                    random_state=RANDOM_STATE,
                ),
                "SVR (RBF)": models_default["SVR (RBF)"],
                f"KNN (k={k_neighbors})": models_default[f"KNN (k={k_neighbors})"],
                "Gradient Boosting": GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=2,
                    learning_rate=0.1,
                    loss="huber",
                    alpha=0.9,
                    random_state=RANDOM_STATE,
                ),
                "Random Forest": models_default["Random Forest"],
            }

        elif loss_type == "mae":
            models = {
                "Ridge": QuantileRegressor(quantile=0.5, alpha=1.0, solver="highs"),
                "ElasticNet": SGDRegressor(
                    loss="epsilon_insensitive",
                    epsilon=0.0,
                    penalty="elasticnet",
                    alpha=0.1,
                    l1_ratio=0.5,
                    max_iter=10000,
                    random_state=RANDOM_STATE,
                ),
                "SVR (RBF)": models_default["SVR (RBF)"],
                f"KNN (k={k_neighbors})": models_default[f"KNN (k={k_neighbors})"],
                "Gradient Boosting": GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=2,
                    learning_rate=0.1,
                    loss="absolute_error",
                    random_state=RANDOM_STATE,
                ),
                "Random Forest": RandomForestRegressor(
                    n_estimators=100,
                    max_depth=3,
                    criterion="absolute_error",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            }

        elif loss_type == "log_cosh":
            # Approximate log-cosh with Huber loss on scaled target
            delta = 1.35  # well-suited when y ~ N(0,1)
            models = {
                "Ridge": HuberRegressor(epsilon=delta, alpha=1.0, max_iter=5000),
                "ElasticNet": SGDRegressor(
                    loss="huber",
                    penalty="elasticnet",
                    alpha=0.1,
                    l1_ratio=0.5,
                    max_iter=10000,
                    random_state=RANDOM_STATE,
                ),
                "SVR (RBF)": models_default["SVR (RBF)"],
                f"KNN (k={k_neighbors})": models_default[f"KNN (k={k_neighbors})"],
                "Gradient Boosting": GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=2,
                    learning_rate=0.1,
                    loss="huber",
                    alpha=0.5,
                    random_state=RANDOM_STATE,
                ),
                "Random Forest": models_default["Random Forest"],
            }

        else:
            raise ValueError(
                f"Unsupported loss_type: {loss_type}. Choose 'squared', 'huber', 'mae', or 'log_cosh'."
            )

        print(f"  Loss type          : {loss_type}")

        # ── 4. Validation: LOO + Repeated 5-Fold (10 repeats) ─────────────────────
        loo = LeaveOneOut()
        records = []

        # We'll store LOO predictions per model for the stratified summary
        loo_preds_dict = {}

        for name, model in models.items():
            print(f"  ▶ {name} ...", end=" ", flush=True)

            # LOO predictions
            loo_preds = np.empty(n_samples)
            for train_idx, test_idx in loo.split(X_scaled):
                # Use scaled target for log_cosh, original otherwise
                y_train = (
                    y_scaled[train_idx] if y_scaler is not None else y[train_idx]
                )
                model.fit(X_scaled[train_idx], y_train)
                pred = model.predict(X_scaled[test_idx])
                # Inverse transform if target was scaled
                if y_scaler is not None:
                    pred = y_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
                loo_preds[test_idx] = pred

            loo_preds_dict[name] = loo_preds.copy()  # store for later

            loo_mae = mean_absolute_error(y, loo_preds)
            loo_rmse = np.sqrt(mean_squared_error(y, loo_preds))
            loo_r2 = r2_score(y, loo_preds)
            loo_medae = median_absolute_error(y, loo_preds)
            loo_rae = loo_mae / mean_absolute_error(y, np.full_like(y, y.mean()))

            # Repeated 5-Fold – manual per-repeat predictions
            kf_mae_vals, kf_rmse_vals, kf_r2_vals = [], [], []
            kf_medae_vals, kf_rae_vals = [], []
            for repeat in range(10):
                kfold = KFold(
                    n_splits=5, shuffle=True, random_state=RANDOM_STATE + repeat
                )
                fold_preds = np.empty(n_samples)
                for train_idx, test_idx in kfold.split(X_scaled):
                    y_train = (
                        y_scaled[train_idx] if y_scaler is not None else y[train_idx]
                    )
                    model.fit(X_scaled[train_idx], y_train)
                    pred = model.predict(X_scaled[test_idx])
                    if y_scaler is not None:
                        pred = y_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
                    fold_preds[test_idx] = pred

                kf_mae_vals.append(mean_absolute_error(y, fold_preds))
                kf_rmse_vals.append(np.sqrt(mean_squared_error(y, fold_preds)))
                kf_r2_vals.append(r2_score(y, fold_preds))
                kf_medae_vals.append(median_absolute_error(y, fold_preds))
                kf_rae_vals.append(
                    mean_absolute_error(y, fold_preds)
                    / mean_absolute_error(y, np.full_like(y, y.mean()))
                )

            kf_mae = np.mean(kf_mae_vals)
            kf_rmse = np.mean(kf_rmse_vals)
            kf_r2 = np.mean(kf_r2_vals)
            kf_medae = np.mean(kf_medae_vals)
            kf_rae = np.mean(kf_rae_vals)

            print("done")
            records.append(
                {
                    "Model": name,
                    "LOO MAE": round(loo_mae, 4),
                    "LOO RMSE": round(loo_rmse, 4),
                    "LOO R²": round(loo_r2, 4),
                    "LOO MedAE": round(loo_medae, 4),
                    "LOO RAE": round(loo_rae, 4),
                    "KFold MAE": round(kf_mae, 4),
                    "KFold RMSE": round(kf_rmse, 4),
                    "KFold R²": round(kf_r2, 4),
                    "KFold MedAE": round(kf_medae, 4),
                    "KFold RAE": round(kf_rae, 4),
                }
            )

        # ── 5. Results table & interpretation ─────────────────────────────────────
        results_df = pd.DataFrame(records).sort_values("LOO MedAE").reset_index(
            drop=True
        )

        print(f"\n{'='*60}")
        print(
            f"📊 PERFORMANCE METRICS  (sorted by LOO MedAE — lower is better)"
        )
        print(f"{'='*60}")
        print(results_df.to_string(index=False))

        # ── Stratified error summary (using LOO predictions of the best MedAE model) ──
        best_model_name = results_df.iloc[0]["Model"]
        best_loo_preds = loo_preds_dict[best_model_name]

        abs_y = np.abs(y)
        bins = [0, 5, 20, np.inf]
        labels = ["small (|y|≤5s)", "medium (5<|y|≤20s)", "large (|y|>20s)"]
        print(f"\n{'='*60}")
        print(
            f"📦 STRATIFIED ERROR (LOO, best MedAE model: {best_model_name})"
        )
        print(f"{'='*60}")
        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            mask = (abs_y >= low) & (abs_y < high) if high != np.inf else (abs_y >= low)
            if mask.sum() == 0:
                print(f"  {labels[i]:<30}: no samples")
                continue
            mae_bin = mean_absolute_error(y[mask], best_loo_preds[mask])
            print(f"  {labels[i]:<30}: count={mask.sum():>3}, MAE={mae_bin:.4f} s")

        # ── 6. Build predictions dataframe ────────────────────────────────────────
        # Re-attach original feature columns (already filtered/aligned via df_filtered)
        predictions_df = df_filtered[cols_to_use].copy().reset_index(drop=True)

        # Add any PRESERVE_COLS that exist in the original df, aligned by index
        preserve_present = [c for c in PRESERVE_COLS if c in df.columns]
        if preserve_present:
            predictions_df = pd.concat(
                [
                    df.loc[df_filtered.index, preserve_present].reset_index(drop=True),
                    predictions_df,
                ],
                axis=1,
            )

        # Add ground truth and per-model LOO predictions
        predictions_df["y_true"] = y
        for model_name, preds in loo_preds_dict.items():
            predictions_df[f"y_pred_LOO_{model_name}"] = preds

        return results_df, predictions_df
