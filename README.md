# covid-19-document-classificaton
NLP project summer semester 2020: Document classifier based on BERT style models using the document classes provided in the LitCovid database

## Topic
Given the rapidly increasing supply of COVID-19 related publications, categorizing the available literature could potentially form an important part in managing the flood of information. Using a pre-trained language model such as BERT [1] and fine-tuning it on task-specific data has shown promising results in the context of document classification [2]. That is why in this project, I plan to build document classification models based on pre-trained BERT-style models. More specifically, I aim at comparing three different variants of transformer models in terms of their classification performance, namely BERT, BioBERT [3] and CovidBERT [4]. While all three models share the same overall architecture and number of parameters, different corpora were used for pre-training. Evidently, the corpora used for the CovidBERT model show the highest relatedness to the COVID-19 subject, followed by the BioBERT model. The original BERT model has no specific relation to the COVID-19 context. With this systematic comparison I hope to investigate the effect of domain-specific corpora used in pre-training on the classification performance in the context of COVID-19 related document classification.


## Data 
To train the classifier, I plan to use the publications listed in the LitCovid [5], [6] database. As of May 26th, metadata of around 16,000 publications, containing abstracts and the listed categories, is publically available through the NCBI Coronavirus API [7] (for instance as jsons). More precisely, the listed categories cover the following classes: Mechanism, General Info, Epidemic Forecasting, Treatment, Transmission, Case Report, Diagnosis, Prevention. Eventually I will exclude one or more classes if they prove to be underrepresented in the data (e.g. <5% of the total number of labels). Although publications can contain multiple class labels, I will restrict the labels to a single label per abstract to reduce the overall complexity of this approach. Moreover, I plan to filter out short abstracts (for instance, abstracts with less than 100 words), since I suspect very short abstracts to be rather noisy than distinctive. Additionally, I plan to apply further preprocessing steps to the abstracts, such as removing stopwords and converting the input to lowercase.


## Evaluation 
In order to measure the classification performance of each model, I aim at using k-fold (k=5) cross-validation, recording the classification reports (including recall, precision and f1-scores for each class) on the validation data for each split. I plan to implement the approach using the FARM [8] and/or spacy [9] libraries in python.

## References
[1]    J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” p. 16.
[2]    A. Adhikari, A. Ram, R. Tang, and J. Lin, “DocBERT: BERT for Document Classification,” ArXiv190408398 Cs, Aug. 2019, Accessed: May 26, 2020. [Online]. Available: http://arxiv.org/abs/1904.08398.
[3]    J. Lee et al., “BioBERT: a pre-trained biomedical language representation model for biomedical text mining,” Bioinformatics, p. btz682, Sep. 2019, doi: 10.1093/bioinformatics/btz682.
[4]    “deepset/covid_bert_base · Hugging Face.” https://huggingface.co/deepset/covid_bert_base (accessed May 26, 2020).
[5]    “LitCovid - NCBI - NLM - NIH.” https://www.ncbi.nlm.nih.gov/research/coronavirus/ (accessed May 26, 2020).
[6]    Q. Chen, A. Allot, and Z. Lu, “Keep up with the latest coronavirus research,” Nature, vol. 579, no. 7798, pp. 193–193, Mar. 2020, doi: 10.1038/d41586-020-00694-1.
[7]    “Health Check – Django REST framework.” https://www.ncbi.nlm.nih.gov/research/coronavirus-api/ (accessed May 26, 2020).
[8]    deepset-ai/FARM. deepset, 2020.
[9]    explosion/spaCy. Explosion, 2020.
