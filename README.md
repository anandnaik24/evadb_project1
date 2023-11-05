# Prompt Iteration

When we use any image generative AI application, we would input a query that sounds intuitive to us. However, the application would not generative the required image because we are not able to provide a descriptive prompt. This descriptive prompt is not intuitive to us. So, we can start with a prompt that is intuitive, search for similar prompts in a database of prompts and and keep refining our prompt till we are satisfied that it does describe the image that we want to generate and then only, input the prompt to the application as generation of images is quite expensive computationally.

## How to use

Step 1: Download the DiffusionDB data - `download_diffusionDB.py`

DiffusionDB provides a feature to download only the prompts and metadata used for generating 2 million of the 14 million images.

Step 2: Set up Fooocus on Google Colab

Choose T4 GPU as the Hardware Accelerator in Google Colab to install and run Fooocus. Follow the commands in [Foocus](https://github.com/lllyasviel/Fooocus#colab) to get a link where the Fooocus app will run.

Step 3: Creating the feature vector for DiffusionDB prompts - `create_diffusionDB_feature_vectors.py`, `feature_vector_function.py`

The feature_vector_function.py is an Abstract Function which takes in a dataframe’s prompt column and returns a feature vector of size 512 for each prompt in the dataframe. It uses the Universal Sentence Encoder to generate the feature vector for each prompt. The feature vector is stored in the table diffusiondb_prompts_features.

Step 4: Comparing the input prompts with prompts in DiffusionDB - `get_similar_prompts.py`

Use input.csv files to load the input prompt in EvaDB which is converted to a feature vector. The euclidean distance of this feature vector and the DiffusionDB feature vectors is then computed and the closest top 100 prompts are stored in the similar_prompts.csv. We can iteratively improve our prompt by borrowing features, vocabulary from the top 100 prompts.
