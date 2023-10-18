import pandas as pd
# Import the EvaDB package
import evadb
import numpy as np

# Connect to EvaDB and get a database cursor for running queries
cursor = evadb.connect().cursor()


#cursor.query("DROP TABLE IF EXISTS prompt_feature_table6;").df()
cursor.query("DROP TABLE IF EXISTS input_prompts_db;").df()
cursor.query("DROP TABLE IF EXISTS input_prompts_db_features6;").df()
relation = cursor.table("prompts_db").select('*').df()
print(relation.head())

print("Loaded prompts into prompts_db Table")

#cursor.query("CREATE TABLE prompts_db (image_name TEXT(100), prompt TEXT(1000))").df()
#cursor.query("CREATE TABLE IF NOT EXISTS features_prompt AS SELECT F2(prompt), prompt FROM prompts_db").df()

# cursor.query("""
#     CREATE INDEX prompts_db_index
#     ON prompts_db (F2(prompt))
#     USING FAISS;
# """).df()

# cursor.query("CREATE TABLE prompt_feature_table6 AS SELECT F2(prompt), prompt FROM prompts_db").df()
# cursor.query("""
#     CREATE INDEX prompt_feature_table_index
#     ON prompt_feature_table6 (features)
#     USING FAISS;
# """).df()

print("created index prompt_feature_table_index")

prompt_db_df = cursor.table("prompt_feature_table6").select('*').df()
print("Relation with index")
print(prompt_db_df.head())
# print(type(relation["prompts_db.prompt"][0]))

#input_prompt = input("Enter your prompt:")
#print(type(input_prompt), "input type")
cursor.query("CREATE TABLE input_prompts_db (prompt TEXT(1000))").df()
cursor.query("LOAD CSV 'input2.csv' INTO input_prompts_db").df()
input_prompt_df = cursor.query("SELECT prompt FROM input_prompts_db").df()
print(input_prompt_df.head())
cursor.query("CREATE TABLE IF NOT EXISTS input_prompts_db_features6 AS SELECT F2(prompt), prompt FROM input_prompts_db").df()
# relation = cursor.table("input_prompts_db_features").select('*').df()

features_column = cursor.query("SELECT features FROM input_prompts_db_features6").df()
print("features_column")
print(features_column.head())

input_feature = features_column["input_prompts_db_features6.features"].loc[0]
print(type(input_feature))
print(input_feature.shape)
#print(input_feature)

prompt_db_df['distance_to_target'] = prompt_db_df['prompt_feature_table6.features'].apply(lambda x: np.linalg.norm(x - input_feature))
sorted_df = prompt_db_df.sort_values(by='distance_to_target')
top_100_min_values = sorted_df.head(100)
print(top_100_min_values)

sorted_df['prompt_feature_table6.prompt'][:100].to_csv("similar_prompts2.csv", index=False)

# test_data = prompt_db_df['prompt_feature_table6.features'].loc[0]
# print(type(test_data))
# print(test_data.shape)
# print(test_data)

print("End of program")
