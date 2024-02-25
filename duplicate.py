import pandas as pd
from sentence_transformers import SentenceTransformer, util

def are_product_names_similar(name1, name2, model):
    # Encode the product names to get their embeddings
    embedding1 = model.encode(name1, convert_to_tensor=True)
    embedding2 = model.encode(name2, convert_to_tensor=True)

    # Calculate cosine similarity between embeddings
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

    similarity_threshold = 0.8

    # Return True if similarity is above the threshold, indicating paraphrased names
    return similarity > similarity_threshold

# Load pre-trained paraphrase detection model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Load the existing dataset from CSV
existing_dataset_path = "product_data.csv"
existing_df = pd.read_csv(existing_dataset_path)

# Example new product name
new_product_name = "Canvas painting wall print featuring the Straw Hat Pirates Luffy, Ace, and Sabo, the trio of brothers, from the anime One Piece."

# Check similarity with each existing product name
existing_df['Similarity'] = existing_df['product_name'].apply(lambda x: are_product_names_similar(x, new_product_name, model))

# Filter the DataFrame to show only the rows where 'Similarity' is True
result_df = existing_df[existing_df['Similarity']]

# Display the results
print(result_df)
