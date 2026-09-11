import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
from utils.parse_coscheduling_runs import CoSchedulingRunsParser

parser = CoSchedulingRunsParser()
data_isolated = parser.parse_isolated("../run_data/experiments_coscheduling_isolated/20260527_083005/data", progress=True)
print("Finished parsing isolated")
data_coscheduled = parser.parse_coscheduled("../run_data/experiments_coscheduling_app_coscheduled/20260527_082956/data", progress=True)
print("Finished parsing app coscheduled")
data_inhib_isolated = parser.parse_inhibitor_isolated("../run_data/experiments_coscheduling_inhib_isolated/20260827_191641/data", progress=True)
print("Finished parsing inhib isolated")
data_inhib_coscheduled = parser.parse_inhibitor_coscheduled_allProfiles("../run_data/experiments_coscheduling_inhib_coscheduled/20260827_191902/data", progress=True)
print("Finished parsing inhib coscheduled")

# Aggregate data_inhib_isolated by Num Nodes, Inhib Message Size, Inhib Wait Time, Inhib Comm Sparsity 
# Group by Num Nodes, Inhib Message Size, Inhib Wait Time, Inhib Comm Sparsity, then calculate mean, median, std for each metric across Run Iterations
data_inhib_isolated_agg = data_inhib_isolated.groupby(['Num Nodes', 'Inhib Message Size', 'Inhib Wait Time (us)', 'Inhib Comm Sparsity']).agg({
    'Inhib App Time Average': ['mean', 'median', 'std'],
    'Inhib MPI Time Average': ['mean', 'median', 'std'],
    'Inhib Total Messages Sent': ['mean', 'median', 'std'],
    'Inhib Total Bytes Sent': ['mean', 'median', 'std']
}).reset_index()

# Flatten column names
data_inhib_isolated_agg.columns = ['Num Nodes', 'Inhib Message Size', 'Inhib Wait Time (us)', 'Inhib Comm Sparsity',
                             'Inhib App Time Average Mean', 'Inhib App Time Average Median', 'Inhib App Time Average Std',
                             'Inhib MPI Time Average Mean', 'Inhib MPI Time Average Median', 'Inhib MPI Time Average Std',
                             'Inhib Total Messages Sent Mean', 'Inhib Total Messages Sent Median', 'Inhib Total Messages Sent Std',
                             'Inhib Total Bytes Sent Mean', 'Inhib Total Bytes Sent Median', 'Inhib Total Bytes Sent Std',
                             ]

data_inhib_isolated_agg


# Aggregate data_isolated by App and Num Nodes
# Group by App and Num Nodes, then calculate mean, median, std for each metric across Run Iterations
data_isolated_agg = data_isolated.groupby(['App', 'Num Nodes']).agg({
    'App Time Average': ['mean', 'median', 'std'],
    'MPI Time Average': ['mean', 'median', 'std'],
    'Total Messages Sent': ['mean', 'median', 'std']
}).reset_index()

# Flatten column names
data_isolated_agg.columns = ['App', 'Num Nodes', 
                             'App Time Average Mean', 'App Time Average Median', 'App Time Average Std',
                             'MPI Time Average Mean', 'MPI Time Average Median', 'MPI Time Average Std',
                             'Total Messages Sent Mean', 'Total Messages Sent Median', 'Total Messages Sent Std']

data_isolated_agg

# Aggregate data_coscheduled by App A, App B, and Num Nodes
# Group by App A, App B, and Num Nodes, then calculate mean, median, std for each metric across Run Iterations
data_coscheduled_agg = data_coscheduled.groupby(['App A', 'App B', 'Num Nodes']).agg({
    'App A Time Average': ['mean', 'median', 'std'],
    'App B Time Average': ['mean', 'median', 'std'],
    'App A MPI Time Average': ['mean', 'median', 'std'],
    'App B MPI Time Average': ['mean', 'median', 'std'],
    'App A Total Messages Sent': ['mean', 'median', 'std'],
    'App B Total Messages Sent': ['mean', 'median', 'std']
}).reset_index()

# Flatten column names
data_coscheduled_agg.columns = ['App A', 'App B', 'Num Nodes',
                                'App A Time Average Mean', 'App A Time Average Median', 'App A Time Average Std',
                                'App B Time Average Mean', 'App B Time Average Median', 'App B Time Average Std',
                                'App A MPI Time Average Mean', 'App A MPI Time Average Median', 'App A MPI Time Average Std',
                                'App B MPI Time Average Mean', 'App B MPI Time Average Median', 'App B MPI Time Average Std',
                                'App A Total Messages Sent Mean', 'App A Total Messages Sent Median', 'App A Total Messages Sent Std',
                                'App B Total Messages Sent Mean', 'App B Total Messages Sent Median', 'App B Total Messages Sent Std']

data_coscheduled_agg

# Aggregate data_inhib coscheduled by App, Num Nodes, Inhib Message Size, Inhib Wait Time (us), Inhib Comm Sparsity
# Group by App, Num Nodes, Inhib Message Size, Inhib Wait Time (us), Inhib Comm Sparsity, Inhib Comm Mode then calculate mean, median, std for each metric across Run Iterations
data_inhib_coscheduled_agg = data_inhib_coscheduled.groupby(['App', 'Num Nodes', 'Inhib Message Size', 'Inhib Wait Time (us)', 'Inhib Comm Sparsity']).agg({
    'App Time Average': ['mean', 'median', 'std'],
    'MPI Time Average': ['mean', 'median', 'std'],
    'Total Messages Sent': ['mean', 'median', 'std'],
    'Inhib App Time Average': ['mean', 'median', 'std'],
    'Inhib MPI Time Average': ['mean', 'median', 'std'],
    'Inhib Total Messages Sent': ['mean', 'median', 'std']
}).reset_index()

# Flatten column names
data_inhib_coscheduled_agg.columns = ['App', 'Num Nodes', 'Inhib Message Size', 'Inhib Wait Time (us)', 'Inhib Comm Sparsity',
                             'App Time Average Mean', 'App Time Average Median', 'App Time Average Std',
                             'MPI Time Average Mean', 'MPI Time Average Median', 'MPI Time Average Std',
                             'Total Messages Sent Mean', 'Total Messages Sent Median', 'Total Messages Sent Std',
                             'Inhib App Time Average Mean', 'Inhib App Time Average Median', 'Inhib App Time Average Std',
                             'Inhib MPI Time Average Mean', 'Inhib MPI Time Average Median', 'Inhib MPI Time Average Std',
                             'Inhib Total Messages Sent Mean', 'Inhib Total Messages Sent Median', 'Inhib Total Messages Sent Std']

data_inhib_coscheduled_agg

# Write processed dataframes to csv files
data_isolated.to_csv("../processed_data/data_isolated.csv")
data_isolated_agg.to_csv("../processed_data/data_isolated_agg.csv")
data_coscheduled.to_csv("../processed_data/data_coscheduled.csv")
data_coscheduled_agg.to_csv("../processed_data/data_coscheduled_agg.csv")
data_inhib_isolated.to_csv("../processed_data/data_inhib_isolated.csv")
data_inhib_isolated_agg.to_csv("../processed_data/data_inhib_isolated_agg.csv")
data_inhib_coscheduled.to_csv("../processed_data/data_inhib_coscheduled.csv")
data_inhib_coscheduled_agg.to_csv("../processed_data/data_inhib_coscheduled_agg.csv")