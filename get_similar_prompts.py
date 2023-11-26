import evadb
import os
import pandas as pd
import numpy as np

# Connect to EvaDB and get a database cursor for running queries
cursor = evadb.connect().cursor()

dir_path = os.path.dirname(os.path.realpath(__file__))

# Drop previously created tables
# cursor.query("DROP TABLE IF EXISTS diffusiondb_prompt").df()
# cursor.query("DROP TABLE IF EXISTS diffusiondb_prompt_features").df()
# cursor.query("DROP INDEX IF EXISTS diffusiondb_prompt_features").df()

# Create the FeatureVectorFunction used to extract feature vector from a prompt
cursor.query(
    f"""
        CREATE FUNCTION IF NOT EXISTS SentenceFeature IMPL "{dir_path}/../../langcache/functions/sentence_feature.py"
    """
).df()

# Create the diffusiondb_prompt to store the prompts
cursor.query("CREATE TABLE IF NOT EXISTS diffusiondb_prompt (prompt TEXT(1000))").df()
cursor.query("LOAD CSV 'output_every_100th_row.csv' INTO diffusiondb_prompt").df()
#cursor.query("LOAD CSV 'output_500_rows.csv' INTO diffusiondb_prompt").df()
prompts = cursor.table("diffusiondb_prompt").select('*').df()
print(prompts.head(5))

cursor.query("CREATE TABLE IF NOT EXISTS diffusiondb_prompt_features AS SELECT SentenceFeature(prompt), prompt FROM diffusiondb_prompt").df()
feature_df = cursor.table("diffusiondb_prompt_features").select('features').df()
print(feature_df.head(5))

cursor.query(
    f"""
        CREATE INDEX IF NOT EXISTS diffusiondb_prompt_features ON diffusiondb_prompt_features (features) USING FAISS
    """
).df()

new_prompt = "black dog playing on mountain, 4k-photo"

# Create the diffusiondb_prompt_features table to store the feature vectors of the prompts in diffusiondb_prompt table
similar_prompts = cursor.query(
    f"""
        SELECT T.prompt, Similarity(SentenceFeature("{new_prompt}"), T.features) FROM
        (SELECT * FROM diffusiondb_prompt_features ORDER BY Similarity(SentenceFeature("{new_prompt}"), features) LIMIT 20) AS T
    """
).df()

print(similar_prompts.head(20))
similar_prompts.to_csv("similar_prompts.csv", index=False)
