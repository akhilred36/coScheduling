import os
import glob
import pandas as pd


class RateLimitParser:
    def __init__(self, directory: str):
        self.directory = directory
        self._bandwidths = None
        self._rate_limits = None

    def parse(self):
        bandwidths_list = []
        rate_limits_list = []

        rate_limit_dirs = sorted(glob.glob(os.path.join(self.directory, "rate_limits_*")))

        for dir_path in rate_limit_dirs:
            run_index_str = os.path.basename(dir_path).replace("rate_limits_", "")
            run_index = int(run_index_str)

            bandwidths_path = os.path.join(dir_path, "bandwidths.csv")
            rate_limits_path = os.path.join(dir_path, "rate_limits.csv")

            if os.path.exists(bandwidths_path):
                df_bandwidths = pd.read_csv(bandwidths_path)
                df_bandwidths["Run Index"] = run_index
                df_bandwidths = df_bandwidths[["Rank A", "Rank B", "Bandwidth (MB/s)", "Run Index"]]
                bandwidths_list.append(df_bandwidths)

            if os.path.exists(rate_limits_path):
                df_rate_limits = pd.read_csv(rate_limits_path)
                df_rate_limits["Run Index"] = run_index
                df_rate_limits = df_rate_limits[["Rank A", "Rank B", "Rate Limit (messages/sec)", "Run Index"]]
                rate_limits_list.append(df_rate_limits)

        if bandwidths_list:
            self._bandwidths = pd.concat(bandwidths_list, ignore_index=True)

        if rate_limits_list:
            self._rate_limits = pd.concat(rate_limits_list, ignore_index=True)

        return self._bandwidths, self._rate_limits

    def get_bandwidths(self) -> pd.DataFrame:
        return self._bandwidths

    def get_rate_limits(self) -> pd.DataFrame:
        return self._rate_limits
