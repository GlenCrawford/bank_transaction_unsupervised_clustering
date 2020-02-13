import glob
import os

import numpy as np
import pandas as pd
import kmodes.kprototypes

import data.merchant_normalization_mapping_expressions

INPUT_DATA_DIRECTORY = 'data/'
INPUT_DATA_FILE_PATTERN = '*.csv'

COLUMN_NAMES = ['Date', 'Amount', 'Merchant', 'Balance After']
COLUMNS_TO_USE = ['Amount', 'Merchant']

NUMBER_OF_CLUSTERS = 7

os.chdir(INPUT_DATA_DIRECTORY)

def load_input_file_data_set(file_path):
  data_frame = pd.read_csv(
    file_path,
    header = None,
    names = COLUMN_NAMES,
    usecols = COLUMNS_TO_USE
  )

  # Files are in reverse chronological order; reverse them.
  data_frame = data_frame.reindex(index = data_frame.index[::-1])

  return data_frame

def load_input_data():
  # Load each file in the input directory into a dataframe.
  data_frames = [load_input_file_data_set(file_path = input_data_file) for input_data_file in glob.glob(INPUT_DATA_FILE_PATTERN)]

  # Merge each files' dataframe into one.
  return pd.concat(
    data_frames,
    axis = 0, # Merge along row axis.
    ignore_index = True
  )

data_frame = load_input_data()

data_frame['Merchant'].replace(
  regex = data.merchant_normalization_mapping_expressions.MERCHANT_NORMALIZATION_MAPPING_EXPRESSIONS,
  inplace = True
)

# Insert the Transaction Type feature.
data_frame['Transaction Type'] = data_frame['Merchant'].map(data.merchant_normalization_mapping_expressions.MERCHANT_TRANSACTION_TYPE_MAPPINGS)

# One-hot encode the Merchant and Transaction Type columns.
categorized_data_frame = pd.get_dummies(
  data_frame,
  columns = ['Merchant', 'Transaction Type'],
  sparse = False
)

model = kmodes.kprototypes.KPrototypes(
  n_clusters = NUMBER_OF_CLUSTERS,
  init = 'Huang',
  verbose = 1
)

clusters = model.fit_predict(
  categorized_data_frame,
  categorical = list(range(1, categorized_data_frame.shape[1])) # Index of columns that contain categorical data. All columns in the dataframe except the first.
)

# Insert the clusters into the dataframe.
data_frame = data_frame.assign(Cluster = pd.Series(clusters).values)

for cluster, cluster_data_frame in data_frame.groupby('Cluster'):
  print('Cluster: ' + str(cluster))
  for index, transaction in cluster_data_frame.iterrows():
    print('  Amount: ' + str(transaction['Amount']) + ', Merchant: ' + str(transaction['Merchant']) + ', Transaction type: ' + str(transaction['Transaction Type']))
