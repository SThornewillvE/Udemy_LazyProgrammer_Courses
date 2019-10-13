# -*- encoding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
C:\Users\SimonThornewill\.spyder-py3\temp.py
"""

import pandas as pd
import io
import requests

# Import Data
url = 'https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/site_data.csv'
s=requests.get(url).content
df=pd.read_csv(io.StringIO(s.decode('utf-8')))

df.columns = ["last_page_id", "next_page_id"]

# ======================================================================================================================
# Analyze data
# ======================================================================================================================

transitions = dict()
row_sums = dict()

# Collect counts
df["count"] = 1
df_counts = df.groupby(by=["last_page_id", "next_page_id"])["count"].count().reset_index()

# Normalise counts
page_counts = df.groupby(by="last_page_id")["count"].count().reset_index()
unique_pages = page_counts["last_page_id"].unique().tolist()

for i in range(len(page_counts)):
    key, val = tuple(page_counts.iloc[i].values)
    row_sums[key] = val

df_counts["last_page_count"] = df_counts["last_page_id"].apply(lambda x: row_sums[x])
df_counts["transitions"] = df_counts["count"]/df_counts["last_page_count"]
df_counts.drop(["last_page_count"], inplace=True, axis=1)

# Get initial state distribution
s = df_counts.query("last_page_id == -1")["transitions"].values
bounce_rates = df_counts.query("next_page_id == 'B'")[["last_page_id", "transitions"]]

# Create transition Matrix
df_prematrix = df_counts[["last_page_id", "next_page_id", "transitions"]]

# Note that this transition matrix should be 13x13 because some transitions that dont happen are missing
# Too lazy to make the real thing tho
Q = df_prematrix.pivot_table(columns = 'next_page_id',
                             index = 'last_page_id',
                             values = 'transitions').values

print(s)
print(bounce_rates)
