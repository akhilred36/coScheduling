import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, roc_curve, precision_recall_curve
from sklearn.model_selection import LeaveOneOut, KFold

RANDOM_STATE = 42


class Classification:
    def __init__(self, class_weight: str = "balanced", scaler_type: str = "standard", TARGET_COL: str = "Good Co-Schedule"):
        """
        Initialize the Classification model trainer.

        Parameters
        ----------
        class_weight : {"balanced"}, default "balanced"
            Class weights to use for imbalanced datasets.
        scaler_type : {"standard", "minmax"}, default "standard"
            The type of scaler to use for feature normalization.
        """
        self.class_weight = class_weight
        self.TARGET_COL = TARGET_COL
        self.PRESERVE_COLS = ["App A", "App B"]
        self.models_ = None
        self.scaler_type = scaler_type
        self.scaler_ = None
        self.feature_cols_ = None

    def train(
        self,
        df: pd.DataFrame,
        selected_features: list,
    ) -> None:
        """
        Train classification models and store them in the class for later prediction.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with features and target column.
        selected_features : list
            List of feature column names to use for training.
        """
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
        print("CLASSIFICATION TRAINING")
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
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_raw)

        n_samples = len(y)
        k_neighbors = min(3, n_samples - 1)

        models_default = {
            "L1": LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=10000, class_weight="balanced"),
            "L2": RidgeClassifier(alpha=1.0),
            "Logistic Regression": LogisticRegression(class_weight="balanced", C=1.0, max_iter=10000),
            "Gaussian Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(
                bootstrap=True,
                class_weight="balanced_subsample",
                max_depth=3,
                n_estimators=100,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        }

        c_values = [0.001, 0.01, 0.1, 1.0, 10.0]
        svm_models = {}
        for c in c_values:
            # svm_models[f"Linear SVM (C={c})"] = SVC(kernel="linear", C=c, class_weight="balanced", probability=True)
            svm_models[f"Linear SVM (C={c})"] = SVC(kernel="linear", C=c, class_weight="balanced", probability=False)

        models = {**models_default, **svm_models}

        print(f"  Models to train    : {len(models)}")

        loo = LeaveOneOut()
        loo_preds_dict = {}

        for name, model in models.items():
            print(f"  ▶ {name} ...", end=" ", flush=True)

            loo_preds = np.empty(n_samples)
            for train_idx, test_idx in loo.split(X_scaled):
                X_train = X_scaled[train_idx]
                y_train = y[train_idx]
                model.fit(X_train, y_train)
                pred = model.predict(X_scaled[test_idx])
                loo_preds[test_idx] = pred

            loo_preds_dict[name] = loo_preds.copy()
            print("done")

        self.models_ = models
        self.scaler_ = scaler
        self.feature_cols_ = cols_to_use

        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"  Models trained: {list(models.keys())}")
        print(f"  Scaler saved: {'Yes' if self.scaler_ else 'No'}")

        results_df = pd.DataFrame([
            {
                "Model": name,
                "LOO Accuracy": round(accuracy_score(y, loo_preds), 4),
            }
            for name, loo_preds in loo_preds_dict.items()
        ]).sort_values("LOO Accuracy", ascending=False).reset_index(drop=True)

        print(f"\n{'='*60}")
        print(f"📊 PERFORMANCE METRICS  (sorted by LOO Accuracy — higher is better)")
        print(f"{'='*60}")
        print(results_df.to_string(index=False))

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        selected_features: list,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Train classification models and evaluate performance using LOO and
        Repeated 5-Fold cross-validation.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with features and target column.
        selected_features : list
            List of feature column names to use for training.

        Returns
        -------
        results_df : pd.DataFrame
            DataFrame with performance metrics for each model.
        predictions_df : pd.DataFrame
            DataFrame with true values and LOO predictions per model.
        """
        self.train(df, selected_features)

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
                model.fit(self.scaler_.transform(df[cols_to_use].dropna().values)[train_idx], y[train_idx])
                pred = model.predict(self.scaler_.transform(df[cols_to_use].dropna().values)[test_idx])
                loo_preds[test_idx] = pred
            loo_preds_dict[name] = loo_preds
            records.append({
                "Model": name,
                "LOO Accuracy": round(accuracy_score(y, loo_preds), 4),
            })

        results_df = pd.DataFrame(records).sort_values("LOO Accuracy", ascending=False).reset_index(drop=True)

        print(f"\n{'='*60}")
        print(f"📊 PERFORMANCE METRICS  (sorted by LOO Accuracy — higher is better)")
        print(f"{'='*60}")
        print(results_df.to_string(index=False))

        best_model_name = results_df.iloc[0]["Model"]
        best_loo_preds = loo_preds_dict[best_model_name]

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
        threshold: float = 0.5,
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
        threshold : float, default 0.5
            Classification threshold for models supporting probability estimates.

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
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_scaled)[:, 1]
                preds = (probs >= threshold).astype(int)
            elif hasattr(model, 'decision_function'):
                scores = model.decision_function(X_scaled)
                preds = (scores >= 0).astype(int)
            else:
                preds = model.predict(X_scaled)
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
            DataFrame with columns: ["model", "Accuracy", "Precision", "Recall", "False Positive Rate", "False Negative Rate", "Area Under Curve"]
        """
        y_true = predictions_df["y_true"].values

        records = []

        pred_cols = [col for col in predictions_df.columns if col.startswith("y_pred_")]

        for col in pred_cols:
            model_name = col.replace("y_pred_", "")
            y_pred = predictions_df[col].values

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)

            tp = ((y_true == 1) & (y_pred == 1)).sum()
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            fpr_score, tpr_score, thresholds = roc_curve(y_true, y_pred)
            auc_score = auc(fpr_score, tpr_score)

            records.append({
                "model": model_name,
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "False Positive Rate": round(fpr, 4),
                "False Negative Rate": round(fnr, 4),
                "Area Under Curve": round(auc_score, 4),
            })

        return pd.DataFrame(records)

    def get_confusion_matrix(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate confusion matrix metrics for each y_pred method in the dataframe.

        Parameters
        ----------
        predictions_df : pd.DataFrame
            DataFrame with 'y_true' column and 'y_pred_*' columns from predict().

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ["model", "TP", "TN", "FP", "FN"]
        """
        y_true = predictions_df["y_true"].values

        records = []

        pred_cols = [col for col in predictions_df.columns if col.startswith("y_pred_")]

        for col in pred_cols:
            model_name = col.replace("y_pred_", "")
            y_pred = predictions_df[col].values

            tp = ((y_true == 1) & (y_pred == 1)).sum()
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()

            records.append({
                "model": model_name,
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
            })

        return pd.DataFrame(records)

    def plot_precision_recall_curve(self, predictions_df: pd.DataFrame) -> None:
        """
        Plot precision-recall curves for each y_pred method in the dataframe.

        Parameters
        ----------
        predictions_df : pd.DataFrame
            DataFrame with 'y_true' column and 'y_pred_*' columns from predict().
        """
        import matplotlib.pyplot as plt

        y_true = predictions_df["y_true"].values

        pred_cols = [col for col in predictions_df.columns if col.startswith("y_pred_")]

        plt.figure(figsize=(8, 6))

        for col in pred_cols:
            model_name = col.replace("y_pred_", "")
            y_pred = predictions_df[col].values

            precision, recall, _ = precision_recall_curve(y_true, y_pred)

            plt.plot(recall, precision, marker='.', label=f"{model_name} (PR-AUC: {auc(recall, precision):.3f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
