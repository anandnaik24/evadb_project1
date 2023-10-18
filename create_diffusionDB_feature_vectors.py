import evadb
import pandas as pd
import numpy as np

# Connect to EvaDB and get a database cursor for running queries
cursor = evadb.connect().cursor()

# Drop previously created tables
cursor.query("DROP TABLE IF EXISTS diffusiondb_prompt;").df()
cursor.query("DROP TABLE IF EXISTS diffusiondb_prompt_features;").df()

# Create the FeatureVectorFunction used to extract feature vector from a prompt
cursor.query("CREATE FUNCTION IF NOT EXISTS FeatureVectorFunction IMPL './feature_vector_function.py';").df()

# Create the diffusiondb_prompt to store the prompts
cursor.query("CREATE TABLE diffusiondb_prompt (image_name TEXT(100), prompt TEXT(1000))").df()
cursor.query("LOAD CSV 'output_every_100th_row.csv' INTO diffusiondb_prompt").df()

# Create the diffusiondb_prompt_features table to store the feature vectors of the prompts in diffusiondb_prompt table
cursor.query("CREATE TABLE IF NOT EXISTS diffusiondb_prompt_features AS SELECT F2(prompt), prompt FROM diffusiondb_prompt").df()