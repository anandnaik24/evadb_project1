import evadb
import pandas as pd
import numpy as np

# Connect to EvaDB and get a database cursor for running queries
cursor = evadb.connect().cursor()

cursor.query("DROP TABLE IF EXISTS input_prompts_db;").df()
cursor.query("DROP TABLE IF EXISTS input_prompt_features;").df()

# Create and load input prompt table
cursor.query("CREATE TABLE input_prompts_db (prompt TEXT(1000))").df()
cursor.query("LOAD CSV 'input1.csv' INTO input_prompts_db").df()

# Create the input prompt features table
cursor.query("CREATE TABLE IF NOT EXISTS input_prompt_features AS SELECT FeatureVectorFunction(prompt), prompt FROM input_prompts_db").df()
features_column = cursor.query("SELECT features FROM input_prompt_features").df()

# Compare the input prompt features with diffusionDB prompts features
input_feature = features_column["input_prompt_features.features"].loc[0]
diffusiondb_prompt_features_df = cursor.table("diffusiondb_prompt_features").select('*').df()
diffusiondb_prompt_features_df['distance_to_target'] = diffusiondb_prompt_features_df['diffusiondb_prompt_features.features'].apply(lambda x: np.linalg.norm(x - input_feature))
sorted_df = prompt_db_df.sort_values(by='distance_to_target')
sorted_df['diffusiondb_prompt_features.prompt'][:100].to_csv("similar_prompts1.csv", index=False)