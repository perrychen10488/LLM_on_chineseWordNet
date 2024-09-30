# word_similarity/similarity.py
import torch
from sentence_transformers import util
from tqdm import tqdm
import pandas as pd

def compute_similarity_st(synwords, model, statistics=True):
    '''
    Extract word embeddings from pre-trained model and compute pair-wise cosine similarity.

    Parameters:
    - synwords: preprocessed synsets from Chinese Wordnet.
    - model: model using sentence transformers framework.
    - statistics: default True, showing descriptive statistics of the result.
    '''

    all_results = []

    for sublist in tqdm(synwords, desc='Processing sublists'):
        if len(sublist) < 2:
            continue

        output = model.encode(sublist)

        if output is None or output.shape[0] == 0:
            print(f"Empty embeddings for sublist: {sublist}, skipping...")
            continue

        cos_sim = util.cos_sim(output, output)
        indices = torch.triu_indices(cos_sim.size(0), cos_sim.size(1), offset=0)
        valid_indices = [(i, j) for i, j in zip(indices[0].tolist(), indices[1].tolist())
                         if i < len(sublist) and j < len(sublist)]

        if valid_indices:
            i_list, j_list = zip(*valid_indices)
            values = cos_sim[i_list, j_list]
            df = pd.DataFrame({'w1': [sublist[i] for i in indices[0].tolist()],
                               'w2': [sublist[i] for i in indices[1].tolist()],
                               'cos_sim': values.tolist()})
            df = df[df['w1'] != df['w2']].reset_index(drop=True)
            all_results.append(df)

    result_df = pd.concat(all_results, ignore_index=True)

    if statistics:
        print(result_df['cos_sim'].describe().apply("{0:.4f}".format))
        return result_df

    return result_df
