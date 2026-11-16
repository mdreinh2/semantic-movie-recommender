# Input provenance

The attached Group 13 presentation specifies The Movies Dataset linked with
MovieLens ratings. This implementation uses the stable MovieLens 20M release.
The three TMDB metadata files come from Rounak Banik's The Movies Dataset,
downloaded through the public Hugging Face mirror below.

| Input | Source |
| --- | --- |
| MovieLens ratings and links | https://grouplens.org/datasets/movielens/20m/ |
| MovieLens archive | https://files.grouplens.org/datasets/movielens/ml-20m.zip |
| Original TMDB dataset | https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset |
| TMDB mirror revision used | https://huggingface.co/datasets/gaygaaa/TheMoviesDataset/tree/287abcb9999f332e13d93b17c9b0352b96875941 |
| TF-IDF implementation reference | https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction |

The metadata mirror provides movies_metadata.csv, keywords.csv, and credits.csv.
MovieLens ratings.csv and links.csv both come from the same official 20M archive.

| File | SHA-256 |
| --- | --- |
| movies_metadata.csv | 37ec322786b136a24604564e55160be75f8937546f11cb1a990076971fa895fd |
| keywords.csv | 428ef94a80255986ad71fd347855702e772ee776841f1e8923a618aa2d87a668 |
| credits.csv | b968359c8e77aebcb7329608378b3b11cc1834726e157aae934bc89f291458ec |
| ratings.csv | 784e85369bbdbbb32e54cbec653a79d087ae1525ffbc5b875de952b72316c00e |
| links.csv | 5c8826f9523d6d004046ef579d42b655c58f7c5d55a868913180cf11e3e31c93 |

The official MovieLens ZIP MD5 was verified as
`cd245b17a1ae2cc31bb14903e1204af3`.

Use the dataset providers' terms for any distribution or publication of their
data. MovieLens specifies research use and requires permission for redistribution.
Its full terms are at https://files.grouplens.org/datasets/movielens/ml-20m-README.html.

Cite F. Maxwell Harper and Joseph A. Konstan, *The MovieLens Datasets: History
and Context*, ACM Transactions on Interactive Intelligent Systems 5(4), 2015,
https://doi.org/10.1145/2827872. Credit Rounak Banik's The Movies Dataset and TMDB
for the movie metadata.
