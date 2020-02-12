import glob
import os

import pandas as pd

import data.merchant_normalization_mapping_expressions

INPUT_DATA_DIRECTORY = 'data/'
INPUT_DATA_FILE_PATTERN = '*.csv'

COLUMN_NAMES = ['Date', 'Amount', 'Merchant', 'Balance After']
COLUMNS_TO_USE = ['Amount', 'Merchant']

# Column "Amount" values are of type: float64.
# Column "Merchant" values are of type: object (really string).

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

unique_merchants = data_frame['Merchant'].unique()
unique_merchants.sort()
print(unique_merchants)
