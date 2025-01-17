# Text Mining 2020/21
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Giovanni1085/AUC_TMCI_2021/main)

Amsterdam University College -- Text Mining -- Winter/Spring 2021.

## Contents

You can use the [Hello World](notebooks/0_HelloWorld.ipynb) notebooks to check that everything is working.

| Week         | Topic           | Materials  |
| ------------- |-------------| -----:|
| 1      | Introduction and Python refresher | <a href='https://docs.google.com/presentation/d/1nnQD0-YW6KMO46iSKRmzs2rT3Ji29GujRmErLg_n8YE/edit?usp=sharing'>slides</a> + notebooks <a href='notebooks/1_Fundamentals.ipynb'>1</a>, <a href='notebooks/1_MoreFundamentals.ipynb'>2</a>, <a href='notebooks/1_EvenMoreFundamentals.ipynb'>3</a>, <a href='notebooks/1_RegularExpressions.ipynb'>4</a>, <a href='notebooks/1_ScientificProgramming.ipynb'>5</a> |
| 2      | Introduction to NLP and NLP pipelines | <a href='https://docs.google.com/presentation/d/159wdX1hSmMZ3qi1VAE5KkxB_oltV5g83ljYSpgIZqX4/edit?usp=sharing'>slides</a> + <a href='notebooks/2_NLP_pipelines.ipynb'>notebook</a> |
| 3      | Language modelling  | <a href='slides/AUC_3_language_models.pdf'>slides</a> + notebooks <a href='notebooks/3_Distributions_in_text.ipynb'>1</a>, <a href='notebooks/3_WordNet.ipynb'>2</a> |
| 4      | Vector space semantics | <a href='slides/AUC_4_vectorSpaceSemantics.pdf'>slides</a> + <a href='notebooks/4_Vector_Semantics.ipynb'>notebook</a> |
| 5      | Word embeddings | <a href='slides/AUC_5_Word_Embeddings.pdf'>slides</a> + <a href='notebooks/5_WordEmbeddings.ipynb'>notebook</a> |
| 6      | Machine learning fundamentals  | <a href='slides/AUC_6_ML_revised.pdf'>slides</a> + <a href='notebooks/6_ML.ipynb'>notebook</a> |
| 7      | Text classification  | <a href='slides/AUC_7_text_classification.pdf'>slides</a> + <a href='notebooks/7_1_Classification.ipynb'>notebook (Scikit-learn)</a>, <a href='notebooks/7_2_PyTorch.ipynb'>notebook (PyTorch)</a> |
| 8      | RNNs and NER  | <a href='slides/AUC_8_RNNs.pdf'>slides</a> + <a href='notebooks/8_1_NER.ipynb'>notebook</a> |
| 9      | Web scraping and APIs  | <a href='notebooks/9_WebScraping_APIs.ipynb'>notebook</a> |
| 10      | Recommender systems  | <a href='slides/AUC_9_Recommender_Systems.pdf'>slides</a> + <a href='notebooks/10_Recommender_Systems.ipynb'>notebook</a> |
| 11      | Creating annotated corpora and sentiment analysis  | <a href='slides/AUC_10_Creating_annotated_corpora.pdf'>slides</a> + <a href='notebooks/11_Sentiment_Analysis.ipynb'>notebook</a> |
| 12      | Clustering and topic modelling  | <a href='slides/AUC_11_Clustering_TM.pdf'>slides</a> + <a href='notebooks/12_Clustering_TopicModelling.ipynb'>notebook</a> |
| 13      | XAI and Ethics  | Selected contents from [this course](https://github.com/Giovanni1085/UvA_AIforSociety_2021) |

### External materials

#### Neural networks

* [Introduction (Stanford's CS231N)](https://cs231n.github.io/neural-networks-1).
* [Optimization 1 (Stanford's CS231N)](https://cs231n.github.io/optimization-1/).
* [Yes you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b) by Andrej Karpathy.
* [Optimization 2 (Stanford's CS231N)](https://cs231n.github.io/optimization-2/).

#### Tutorials

* [SpaCy 101](https://spacy.io/usage/spacy-101).
* [scikit-learn tutorials](https://scikit-learn.org/stable/tutorial/index.html).
* [PyTorch 60m blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).
* [PyTorch text classification with torchtext](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html).

## Group projects

See the [projects folder](/projects) for info.

### Project outcomes

* Tim Holthuijsen, Sergio Kazatzidis, Emilia Chammas, [NLPChef - An NLP model for cooking recipe creation](https://github.com/timholthuijsen/NLPChef)
* Lisa van Gelderen, Alexia Muresan, Zoë Prins, Victor van der Sman, [Gender Bias in Fairy Tales](https://github.com/lisavangelderen/FairyTale)
* Sarah de Jong, Tom Klein Tijssink, Lukas Busch, [Exploring Different Machine Learning Approaches To Generate Song Lyrics](https://github.com/Brahex/text-mining-final-project)
* Yiyang Cheng, Amaan Syed, Tori Baral, [Explaining the qualitative differences between different classifiers on cyberbullying datasets](https://github.com/cyymbanzakongo/TextMiningYAT)
* Ilai Bachrach, Marc Oliveau, [Lyric-based recommendation for music playlists](https://github.com/MarcOliveau/ContentBased_RecommenderSystem)
* Berke Filiz, Yuval Goren, AnneLouise de Boer, [Recipe Generator](https://github.com/berkefiliz/text-mining-project/tree/Final_branch)

## Set-up

1. Clone the repository locally: `git clone https://github.com/Giovanni1085/AUC_TMCI_2021.git`
2. Get updates (from time to time): `git pull`
3. Create a conda environemnt: `conda create -n myenv python=3.7 anaconda` (where `myenv` is the envirnoment name)
4. Activate it: `conda activate myenv`
5. Install packages (see the `requirements.txt` file), e.g. `conda install pandas`
6. Launch a Jupyter notebook: `jupyter notebook`

* [More on conda enviroments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
* [Conda cheatsheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
* [Getting started with Jupyter notebooks](https://medium.com/codingthesmartway-com-blog/getting-started-with-jupyter-notebook-for-python-4e7082bd5d46)
* [On using git and GitHub for version control](https://alan-turing-institute.github.io/rsd-engineeringcourse/ch02git)

Alternatively, use [Binder](https://mybinder.org) (link above).

**A more detailed [guide to setup your environment](setup.md), with multiple options.**

## Credits

* The [previous-year edition](https://github.com/Giovanni1085/AUC_TMCI_2019) of this course.
* Michael Repplinger, who ran the 2018/19 edition and Gianluca Lebani, who ran the 2017/18 edition.
* Giovanni Colavizza and Matteo Romanello, Applied Data Analysis course for the Oxford Digitial Humanities Summer School [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3352830.svg)](https://doi.org/10.5281/zenodo.3352830)
* James Hetherington and Giovanni Colavizza, [Research Software Engineering with Python](https://alan-turing-institute.github.io/rsd-engineeringcourse/)

## License

Everything in this repository which is not already attributed to someone else is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 
