**Towards Malay Abbreviation Disambiguation: Corpus and Unsupervised Model**

 word2vec method,please see this paper[[ResearchGate](https://www.researchgate.net/publication/346792753_Unsupervised_Abbreviation_Disambiguation_Contextual_disambiguation_using_word_embeddings/link/5fd0ec68a6fdcc697bf09241/download)] in detail.

* data folder includes Malay dataset we created and SDU@AAAI-22-Shared_Task_2 dataset.


* Supervised_model folder includes ADBCMM (siamese) and ADBCMM (concat) models.


* main.py is our method for abbreviation disambiguation. It includes the first order perplexity, the second order perplexity and the third order perplexity.
* It is worth noting that weighted_fusion.py is our weighted fusion method.
* scorer.py is that SDU@AAAI-22-Shared_Task_2 provide official evaluation method, on this basis, we make improvements for convenient using.

we used the MLP (pre-trained model), more details please see [Hugging Face – The AI community building the future.](https://huggingface.co/):

```
bert-base-multilingual-cased
malay-huggingface/bert-base-bahasa-cased
dccuchile/bert-base-spanish-wwm-uncased
bertin-project/bertin-roberta-base-spanish
bert-base-uncased
roberta-base
dbmdz/bert-base-french-europeana-cased
```

* tip for our method:

1. After we find dataset, we need to run main.py firstly to make the json weighted file.
2. After we get weighted file, we run weighted_fusion.py to get final result.

