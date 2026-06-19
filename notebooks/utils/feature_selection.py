import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge


class FeatureSelection:
    """Feature selection for co-scheduled app timing data."""

    def __init__(self, TARGET_COL: str = "App A Co-Scheduled MPI Time with App B"):
        self.TARGET_COL = TARGET_COL
        self.all_selections = None
        self.feature_cols = None
        self.df = None

    def get_feature_matrix(self, df: pd.DataFrame, k: int = 8, random_state: int = 42):
        """
        Run all feature selection methods and store results.

        Args:
            df: Input dataframe
            k: Number of top features to select per method
            random_state: Random state for reproducibility

        Returns:
            dict: Feature selection results from all methods
        """
        PRESERVE_COLS = ["App A", "App B"]

        self.df = df
        print(f"Using dataframe: {df.shape[0]} rows, {df.shape[1]} columns")

        # Identify columns
        feature_cols = [
            c for c in df.columns
            if c not in PRESERVE_COLS + [self.TARGET_COL]
        ]
        self.feature_cols = feature_cols
        print(f"  Preserved columns : {PRESERVE_COLS}")
        print(f"  Target column     : {self.TARGET_COL}")
        print(f"  Feature columns   : {len(feature_cols)}")

        always_include = "App A Isolated MPI Time"
        if always_include in df.columns and always_include in feature_cols:
            feature_cols.remove(always_include)
            print(f"  (Always including '{always_include}' separately)")

        # Normalize features
        X_raw = df[feature_cols].values
        y = df[self.TARGET_COL].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        print(f"\nStandardization complete (StandardScaler).")
        print(f"  X shape: {X_scaled.shape}, y shape: {y.shape}")

        # Run all selection methods
        all_selections = {
            "Correlation": self._select_by_correlation(df, feature_cols, k),
            "MutualInfo": self._select_by_mutual_info(X_scaled, y, feature_cols, k, random_state),
            "RandomForest": self._select_by_random_forest(X_scaled, y, feature_cols, k, random_state),
            "Lasso": self._select_by_lasso(X_scaled, y, feature_cols, k, random_state),
            "RFE_Ridge": self._select_by_rfe(X_scaled, y, feature_cols, k),
        }

        if always_include in df.columns:
            for method, features in all_selections.items():
                if always_include not in features:
                    features.append(always_include)

        self.all_selections = all_selections

        return all_selections

    def get_consensus(self) -> pd.DataFrame:
        """
        Compute consensus from previously stored feature selections.

        Returns:
            pd.DataFrame: Consensus results with feature rankings
        """
        if self.all_selections is None or self.feature_cols is None:
            raise RuntimeError(
                "No feature selections computed. Call get_feature_matrix() first."
            )

        print("\n" + "=" * 60)
        print("CONSENSUS RESULTS")
        print("=" * 60)

        counter = Counter()
        for method, features in self.all_selections.items():
            counter.update(features)

        consensus_df = pd.DataFrame(
            counter.most_common(),
            columns=["Feature", "Methods_Count"]
        )
        consensus_df["Methods"] = consensus_df["Feature"].apply(
            lambda f: [m for m, feats in self.all_selections.items() if f in feats]
        )

        n_methods = len(self.all_selections)

        unanimous = consensus_df[consensus_df["Methods_Count"] == n_methods]
        print(f"\n Features selected by ALL {n_methods} methods ({len(unanimous)} features):")
        if unanimous.empty:
            print("   (none — consider lowering k or the consensus threshold)")
        else:
            for _, row in unanimous.iterrows():
                print(f"   • {row['Feature']}")

        majority = consensus_df[
            (consensus_df["Methods_Count"] >= n_methods / 2) &
            (consensus_df["Methods_Count"] < n_methods)
        ]
        print(f"\n Features selected by MAJORITY (≥{n_methods//2 + 1}) but not all ({len(majority)} features):")
        for _, row in majority.iterrows():
            print(f"   • {row['Feature']}  (picked by {row['Methods_Count']}/{n_methods}: {row['Methods']})")

        print("\n Full ranked consensus table:")
        print(consensus_df[["Feature", "Methods_Count"]].to_string(index=False))

        return consensus_df

    def _select_by_correlation(self, df: pd.DataFrame, feature_cols: list, k: int) -> list:
        """Pearson correlation between each feature and the target."""
        print("\n[1] Pearson Correlation ...")
        corr = df[feature_cols + [self.TARGET_COL]].corr()[self.TARGET_COL].drop(self.TARGET_COL)
        top = corr.abs().nlargest(k).index.tolist()
        print(f"    Top {k}: {top}")
        return top

    def _select_by_mutual_info(
        self, X: np.ndarray, y: np.ndarray, feature_cols: list, k: int, random_state: int
    ) -> list:
        """Mutual information regression."""
        print("\n[2] Mutual Information Regression ...")
        mi_scores = mutual_info_regression(X, y, random_state=random_state)
        mi_series = pd.Series(mi_scores, index=feature_cols)
        top = mi_series.nlargest(k).index.tolist()
        print(f"    Top {k}: {top}")
        return top

    def _select_by_random_forest(
        self, X: np.ndarray, y: np.ndarray, feature_cols: list, k: int, random_state: int
    ) -> list:
        """Random Forest feature importances."""
        print("\n[3] Random Forest Feature Importance ...")
        rf = RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
        rf.fit(X, y)
        rf_series = pd.Series(rf.feature_importances_, index=feature_cols)
        top = rf_series.nlargest(k).index.tolist()
        print(f"    Top {k}: {top}")
        return top

    def _select_by_lasso(
        self, X: np.ndarray, y: np.ndarray, feature_cols: list, k: int, random_state: int
    ) -> list:
        """Lasso (L1) regression with alpha tuning."""
        print("\n[4] Lasso (L1 Regularization) ...")

        best_alpha = None
        best_top = []
        best_diff = float("inf")

        for alpha in np.logspace(-4, 1, 60):
            lasso = Lasso(alpha=alpha, max_iter=10000, random_state=random_state)
            lasso.fit(X, y)
            nonzero = np.sum(lasso.coef_ != 0)
            diff = abs(nonzero - k)
            if diff < best_diff:
                best_diff = diff
                best_alpha = alpha
                coef_series = pd.Series(np.abs(lasso.coef_), index=feature_cols)
                best_top = coef_series[coef_series > 0].nlargest(k).index.tolist()

        print(f"    Best alpha: {best_alpha:.6f}  (non-zero features ≈ {len(best_top)})")
        print(f"    Top {k}: {best_top}")
        return best_top

    def _select_by_rfe(self, X: np.ndarray, y: np.ndarray, feature_cols: list, k: int) -> list:
        """Recursive Feature Elimination using Ridge."""
        print("\n[5] Recursive Feature Elimination (RFE with Ridge) ...")
        ridge = Ridge()
        rfe = RFE(estimator=ridge, n_features_to_select=k)
        rfe.fit(X, y)
        top = [feature_cols[i] for i, s in enumerate(rfe.support_) if s]
        print(f"    Top {k}: {top}")
        return top
