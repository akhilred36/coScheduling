import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor, SGDRegressor, QuantileRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, KFold

RANDOM_STATE = 42


class Regression:
    def __init__(self, loss_type: str = "squared", scaler_type: str = "standard", TARGET_COL: str = "App A Co-Scheduled MPI Time with App B"):
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
        self.TARGET_COL = TARGET_COL
        self.PRESERVE_COLS = ["App A", "App B"]
        self.models_ = None
        self.scaler_type = scaler_type
        self.scaler_ = None
        self.y_scaler_ = None
        self.feature_cols_ = None

    def train(
        self,
        df: pd.DataFrame,
        selected_features: list,
        loss_type: str = None,
    ) -> None:
        """
        Train regression models and store them in the class for later prediction.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with features and target column.
        selected_features : list
            List of feature column names to use for training.
        loss_type : {"squared", "huber", "mae", "log_cosh"}, optional
            Override the loss type set during initialization.
        """
        if loss_type is None:
            loss_type = self.loss_type

        TARGET_COL = self.TARGET_COL
        PRESERVE_COLS = self.PRESERVE_COLS

        cols_to_use = [c for c in selected_features if c not in PRESERVE_COLS]
        excluded = [c for c in selected_features if c in PRESERVE_COLS]

        missing = [c for c in cols_to_use if c not in df.columns]
        if missing:
            raise ValueError(f"These requested features are not in the dataframe: {missing}")
        if TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe.")

        df_filtered = df[cols_to_use + [TARGET_COL]].dropna().copy()

        print(f"\n{'='*60}")
        print("REGRESSION TRAINING")
        print(f"{'='*60}")
        print(f"  Features used      : {len(cols_to_use)}  → {cols_to_use}")
        print(f"  Columns excluded   : {excluded if excluded else 'none'}")
        print(f"  Samples after dropna: {len(df_filtered)}")
        print(f"  Target             : {TARGET_COL}")

        X_raw = df_filtered[cols_to_use].values
        y = df_filtered[TARGET_COL].values.astype(float)

        if (self.scaler_type == "standard"):
            scaler = StandardScaler()
        elif (self.scaler_type == "minmax"):
            scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_raw)

        if loss_type == "log_cosh":
            y_scaler = StandardScaler()
            y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
            print("  (target scaled for log_cosh)")
        else:
            y_scaler = None
            y_scaled = y

        n_samples = len(y)
        k_neighbors = min(3, n_samples - 1)

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
            delta = 1.35
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
            delta = 1.35
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

        loo = LeaveOneOut()
        loo_preds_dict = {}

        for name, model in models.items():
            print(f"  ▶ {name} ...", end=" ", flush=True)

            loo_preds = np.empty(n_samples)
            for train_idx, test_idx in loo.split(X_scaled):
                y_train = (
                    y_scaled[train_idx] if y_scaler is not None else y[train_idx]
                )
                X_train = X_scaled[train_idx]
                model.fit(X_train, y_train)
                pred = model.predict(X_scaled[test_idx])
                if y_scaler is not None:
                    pred = y_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
                loo_preds[test_idx] = pred

            loo_preds_dict[name] = loo_preds.copy()
            print("done")

        self.models_ = models
        self.scaler_ = scaler
        self.y_scaler_ = y_scaler
        self.feature_cols_ = cols_to_use

        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"  Models trained: {list(models.keys())}")
        print(f"  Scaler saved: {'Yes' if self.scaler_ else 'No'}")
        print(f"  Target scaler saved: {'Yes' if self.y_scaler_ else 'No'}")

        results_df = pd.DataFrame([
            {
                "Model": name,
                "LOO MAE": round(mean_absolute_error(y, loo_preds), 4),
                "LOO RMSE": round(np.sqrt(mean_squared_error(y, loo_preds)), 4),
                "LOO R²": round(r2_score(y, loo_preds), 4),
            }
            for name, loo_preds in loo_preds_dict.items()
        ]).sort_values("LOO RMSE").reset_index(drop=True)

        print(f"\n{'='*60}")
        print(f"📊 PERFORMANCE METRICS  (sorted by LOO RMSE — lower is better)")
        print(f"{'='*60}")
        print(results_df.to_string(index=False))

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
        self.train(df, selected_features, loss_type)

        TARGET_COL = self.TARGET_COL
        PRESERVE_COLS = self.PRESERVE_COLS
        cols_to_use = self.feature_cols_
        y = df[cols_to_use + [TARGET_COL]].dropna()[TARGET_COL].values.astype(float)

        loo_preds_dict = {}
        records = []

        for name, model in self.models_.items():
            loo_preds = np.empty(len(y))
            loo = LeaveOneOut()
            for train_idx, test_idx in loo.split(self.scaler_.transform(df[cols_to_use].dropna().values)):
                y_train = (
                    self.y_scaler_.transform(y[train_idx].reshape(-1, 1)).ravel()
                    if self.y_scaler_ is not None else y[train_idx]
                )
                model.fit(self.scaler_.transform(df[cols_to_use].dropna().values)[train_idx], y_train)
                pred = model.predict(self.scaler_.transform(df[cols_to_use].dropna().values)[test_idx])
                if self.y_scaler_ is not None:
                    pred = self.y_scaler_.inverse_transform(pred.reshape(-1, 1)).ravel()
                loo_preds[test_idx] = pred
            loo_preds_dict[name] = loo_preds
            records.append({
                "Model": name,
                "LOO MAE": round(mean_absolute_error(y, loo_preds), 4),
                "LOO RMSE": round(np.sqrt(mean_squared_error(y, loo_preds)), 4),
                "LOO R²": round(r2_score(y, loo_preds), 4),
                "LOO MedAE": round(median_absolute_error(y, loo_preds), 4),
                "LOO RAE": round(mean_absolute_error(y, loo_preds) / mean_absolute_error(y, np.full_like(y, y.mean())), 4),
            })

        results_df = pd.DataFrame(records).sort_values("LOO MedAE").reset_index(drop=True)

        print(f"\n{'='*60}")
        print(f"📊 PERFORMANCE METRICS  (sorted by LOO MedAE — lower is better)")
        print(f"{'='*60}")
        print(results_df.to_string(index=False))

        best_model_name = results_df.iloc[0]["Model"]
        best_loo_preds = loo_preds_dict[best_model_name]

        abs_y = np.abs(y)
        bins = [0, 5, 20, np.inf]
        labels = ["small (|y|≤5s)", "medium (5<|y|≤20s)", "large (|y|>20s)"]
        print(f"\n{'='*60}")
        print(f"📦 STRATIFIED ERROR (LOO, best MedAE model: {best_model_name})")
        print(f"{'='*60}")
        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            mask = (abs_y >= low) & (abs_y < high) if high != np.inf else (abs_y >= low)
            if mask.sum() == 0:
                print(f"  {labels[i]:<30}: no samples")
                continue
            mae_bin = mean_absolute_error(y[mask], best_loo_preds[mask])
            print(f"  {labels[i]:<30}: count={mask.sum():>3}, MAE={mae_bin:.4f} s")

        predictions_df = df[cols_to_use].dropna().copy().reset_index(drop=True)
        preserve_present = [c for c in PRESERVE_COLS if c in df.columns]
        if preserve_present:
            predictions_df = pd.concat(
                [df.loc[df[cols_to_use + [TARGET_COL]].dropna().index, preserve_present].reset_index(drop=True), predictions_df],
                axis=1,
            )
        predictions_df["y_true"] = y
        for model_name, preds in loo_preds_dict.items():
            predictions_df[f"y_pred_LOO_{model_name}"] = preds

        return results_df, predictions_df

    def predict(
        self,
        df: pd.DataFrame,
        selected_features: list,
        model_names: list = None,
    ) -> pd.DataFrame:
        """
        Make predictions using the trained models.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with features and target column.
        selected_features : list
            List of feature column names to use for prediction.
        model_names : list, optional
            Specific model names to predict with. If None, uses all trained models.

        Returns
        -------
        predictions_df : pd.DataFrame
            DataFrame with true values (if available) and predictions per model.
        """
        if self.models_ is None:
            raise RuntimeError("No trained models found. Call train() first.")

        TARGET_COL = self.TARGET_COL
        PRESERVE_COLS = self.PRESERVE_COLS
        cols_to_use = [c for c in selected_features if c not in PRESERVE_COLS]

        missing = [c for c in cols_to_use if c not in df.columns]
        if missing:
            raise ValueError(f"These requested features are not in the dataframe: {missing}")
        if TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe.")

        df_filtered = df[cols_to_use + [TARGET_COL]].dropna().copy()
        X_raw = df_filtered[cols_to_use].values
        y = df_filtered[TARGET_COL].values.astype(float)

        X_scaled = self.scaler_.transform(X_raw)

        predictions_df = df_filtered[cols_to_use].copy().reset_index(drop=True)

        preserve_present = [c for c in PRESERVE_COLS if c in df.columns]
        if preserve_present:
            predictions_df = pd.concat(
                [df.loc[df_filtered.index, preserve_present].reset_index(drop=True), predictions_df],
                axis=1,
            )

        predictions_df["y_true"] = y
        if "App A Isolated MPI Time" in df.columns:
            predictions_df["App A Isolated MPI Time"] = df.loc[df_filtered.index, "App A Isolated MPI Time"].values
            predictions_df["App A Co-Scheduled MPI Time with App B"] = df.loc[df_filtered.index, TARGET_COL].values

        models_to_use = self.models_ if model_names is None else {k: v for k, v in self.models_.items() if k in model_names}

        for name, model in models_to_use.items():
            preds = model.predict(X_scaled)
            if self.y_scaler_ is not None:
                preds = self.y_scaler_.inverse_transform(preds.reshape(-1, 1)).ravel()
            predictions_df[f"y_pred_{name}"] = preds

        return predictions_df

    def get_error_metrics(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate error metrics for each y_pred method in the dataframe.

        Parameters
        ----------
        predictions_df : pd.DataFrame
            DataFrame with 'y_true' column and 'y_pred_*' columns from predict().

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ["model", "mse", "mae", "r2", "rmse", "mape"]
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        y_true = predictions_df["y_true"].values

        records = []

        if "App A Isolated MPI Time" in predictions_df.columns:

            y_pred = predictions_df["App A Isolated MPI Time"].values
            if ("Slowdown" in self.TARGET_COL):
                y_pred = np.ones_like(y_pred) * 1.0
            
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mse)

            y_true_nonzero = y_true[y_true != 0]
            y_pred_nonzero = y_pred[y_true != 0]
            if len(y_true_nonzero) > 0:
                mape = np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)) * 100
            else:
                mape = 0.0

            records.append({
                "model": "App A Isolated MPI Time",
                "mse": round(mse, 4),
                "mae": round(mae, 4),
                "r2": round(r2, 4),
                "rmse": round(rmse, 4),
                "mape": round(mape, 4),
            })

        pred_cols = [col for col in predictions_df.columns if col.startswith("y_pred_")]

        for col in pred_cols:
            model_name = col.replace("y_pred_", "")
            y_pred = predictions_df[col].values

            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mse)

            y_true_nonzero = y_true[y_true != 0]
            y_pred_nonzero = y_pred[y_true != 0]
            if len(y_true_nonzero) > 0:
                mape = np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)) * 100
            else:
                mape = 0.0

            records.append({
                "model": model_name,
                "mse": round(mse, 4),
                "mae": round(mae, 4),
                "r2": round(r2, 4),
                "rmse": round(rmse, 4),
                "mape": round(mape, 4),
            })

        return pd.DataFrame(records)
