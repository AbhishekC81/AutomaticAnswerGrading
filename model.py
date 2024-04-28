from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import preprocess

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

client = OpenAI()


def get_embedding(text):
    model = "text-embedding-3-small"
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def calculate_cosine_similarity(embedding1, embedding2):
    cos_sim = cosine_similarity([embedding1], [embedding2])
    return cos_sim[0][0]


def run_model(sample_answer, df, total_marks):
    # Preprocess the sample answer
    sample_answer_preprocessed = preprocess.preprocess_text(sample_answer)

    # Get embedding vector for the preprocessed sample answer
    sample_embedding = get_embedding(sample_answer_preprocessed)

    # Preprocess the text in the DataFrame
    df['text_preprocessed'] = df['text'].apply(preprocess.preprocess_text)

    # Get embedding vectors for preprocessed text
    df['embedding'] = df['text_preprocessed'].apply(get_embedding)

    # Calculate cosine similarity between each embedding vector in the DataFrame and the sample_embedding
    df['cos_score'] = df.apply(lambda row: calculate_cosine_similarity(row['embedding'], sample_embedding), axis=1)

    # Calculate the final score by multiplying the total_marks with cos_score
    df['final_score'] = round(total_marks * df['cos_score'], 1)

    # Drop unnecessary columns
    df = df[['text', 'final_score']]

    return df
