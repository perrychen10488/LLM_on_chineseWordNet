# LLM on Chinese WordNet experiment

This notebook is for testing different embedding models on Chinese Wordnet synsets.

Usagae:
- import relevant packages and install Chinese Wordnet data
- initiate a model from sentence-transformers, and specifiy the model name in `compute_similarity_st()`
- the function will return pair-wise cosine simialrity measures for the synsets and store them in a data frame
- generate boxplots with results from different models by `generate_boxplot()`
