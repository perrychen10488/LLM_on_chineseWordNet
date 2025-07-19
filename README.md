# LLM on Chinese WordNet experiment

This tool is for testing different embedding models on Chinese Wordnet synsets. User can have a quick check how well a model perform on Chinese words when conducting sematnic research on word level.

## Usagae:
- import relevant packages and install Chinese Wordnet data [^1]
- initiate a model from sentence-transformers, and specifiy the model name in `compute_similarity_st()`
- the function will return pair-wise cosine simialrity measures for the synsets and store them in a data frame.

- generate boxplots with results from different models by `generate_boxplot()`

## Result
![image](https://github.com/perrychen10488/LLM_on_chineseWordNet/blob/master/img/output.png)



[^1]: The synsets are extracted from Chinese Wordnet, and is further preprocessed to keep non-empty synsets. For more information on the term of use and construction of Chinese Wordnet, please refer to [Chinese Wordnet](https://lopentu.github.io/CwnWeb/#home)