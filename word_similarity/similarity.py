# word_similarity/similarity.py
import torch
from sentence_transformers import util
from tqdm import tqdm
import pandas as pd
import numpy as np
import sentence_transformers
import transformers
import torch
import re
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# for transformer models
# compute embeddings from transformer model
def get_word_embedding(words, tokenizer, model, batch_size=1000):
    from tqdm import tqdm
    model.to(device)
    ''' text = text to encode in list
    '''
    word_embeddings = {}
    
    # process words in batch
    def process_batch(batch):
        # encode words
        encoding = tokenizer.batch_encode_plus(
        batch,
        padding = True,
        return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)

        # get last hidden states
        with torch.no_grad():
            output = model(input_ids, attention_mask = encoding['attention_mask'].to(device), output_hidden_states=True,
                           )
            # emb = model.bert.embeddings.word_embeddings(input_ids)
            last_hidden_state = output.hidden_states[-1]
            



        # last_hidden_state = output.hidden_states[-1]

        # compute meaning embeddings for word in text
        # Convert token IDs to tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        # Prepare to aggregate embeddings


        # Iterate over each word and its corresponding tokens
        for batch_index, (word, token_ids) in enumerate(zip(words, input_ids)):
            word_token_embeddings = []
            token_ids = token_ids.tolist()  # Convert tensor to list

            # Get embeddings for each token
            for token_index in range(len(token_ids)):
                token_id = token_ids[token_index]
                if token_id == tokenizer.pad_token_id:
                    continue  # Skip padding tokens

                # Get the token embedding from the hidden states
                word_token_embeddings.append(last_hidden_state[batch_index][token_index].cpu().numpy())
                # word_token_embeddings.append(emb[batch_index][token_index].cpu().numpy())

            # Compute the mean of the token embeddings for the word
            if word_token_embeddings:
                word_embeddings[word] = torch.mean(torch.tensor(word_token_embeddings), dim=0).numpy()
                # word_embeddings[word] = torch.sum(torch.tensor(word_token_embeddings), dim=0).numpy()

        # Clear cache to free memory
        # del encoding, input_ids, output, last_hidden_state
        # torch.cuda.empty_cache()

    # process batch
    for i in range(0, len(words), batch_size):
        batch = words[i:i+batch_size]
        process_batch(batch)
        # print(f"Processed batch {i // batch_size + 1}")


    return word_embeddings

# compute pair-wise cosine simialarity
def compute_similarity(synwords, tokenizer, model, statistics=True):
    
    all_results = []

    # generate embeddings
    for i in tqdm(synwords,desc='Processing sublists'):
        if len(i) < 2:
          continue

        synwords_emb = get_word_embedding(i, tokenizer,model)

        if not synwords_emb: # hancle empty embeddings
            continue

        # compute cosine simialrity
        cos_sim = util.cos_sim(torch.tensor(np.array(list(synwords_emb.values()))),
                    torch.tensor(np.array(list(synwords_emb.values()))))

        # keep only right triangle
        indices = torch.triu_indices(cos_sim.size(0), cos_sim.size(1), offset=0) # upper triangle part

        values = cos_sim[indices[0], indices[1]]
        # values.append(cos_sim[indices[0], indices[1]])


        # df = pd.DataFrame(values, columns = [list(synwords_emb.keys())[i] for i in indices[0].tolist()],
        #                   index = [list(synwords_emb.keys())[i] for i in indices[0].tolist()])
        df = pd.DataFrame({'w1': [list(synwords_emb.keys())[i] for i in indices[0].tolist()],
                        'w2': [list(synwords_emb.keys())[i] for i in indices[1].tolist()],
                        'cos_sim': values})
        df.index.name = None

        # reset index and convert to dataframe
        df = df[df['w1'] != df['w2']] # remove duplicated
        df = df.reset_index(drop=True)

        all_results.append(df)

        # Clean up memory by deleting variables no longer needed
        # uncomment in google colab
        # del synwords_emb, embeddings_tensor, cos_sim, df
        # torch.cuda.empty_cache()

    result_df = pd.concat(all_results, ignore_index=True)

    if statistics:
        print(result_df['cos_sim'].describe().apply("{0:.4f}".format))
        return result_df

    return result_df