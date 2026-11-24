# Sentiment analysis

Binary sentiment classification on the supplied Twitter files. The script trains the representations required by the Practical Machine Learning brief and evaluates them on the held-out test files.

## Representations

| Pipeline | What it builds | Classifier |
|---|---|---|
| `lexicon` | Hand-written positive and negative word lists | Sign of the word score |
| `bow` | Unigram and bigram counts | L2 logistic regression |
| `tf` | L2-normalised term frequency | L2 logistic regression |
| `idf` | Binary term presence weighted by smoothed IDF | L2 logistic regression |
| `tfidf` | Sublinear TF-IDF | L2 logistic regression |
| `word2vec` | Mean of skip-gram Word2Vec word vectors | L2 logistic regression |

The lexicon row is a baseline. The other five rows are the required representations. Logistic regression uses the same model family throughout, so differences come from the representation. `C` is chosen on a stratified 10% validation fold by macro F1, then the chosen model is refit on the full training sample.

## Data

Run the commands below from this `SentimentAnalysis` folder. The script reads `Sentiment Analysis Twitter.zip` in the parent `jointProject` folder. Labels come from the filenames.

- Training pool: `train_Pos.en` (399,000) and `train_Neg.en` (399,000)
- Test set, used only for the final numbers: `ttestPos.en` (1,000) and `ttestNeg.en` (998)

The default run draws a stratified random sample of 50,000 tweets per class from the training pool (`--per-class` changes this). The test files are never sampled and never used for fitting.

## Run

Open `sentiment_analysis.ipynb`. That notebook is the project: data, preprocessing, vectorisation, models, tuning, pruning, and the comparison figures.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name sentiment-analysis --display-name "Python (Sentiment Analysis)"
jupyter notebook sentiment_analysis.ipynb
```

In Cursor, select the kernel **Python (Sentiment Analysis)** and run all cells. The first run downloads the pretrained GloVe Twitter vectors.

## Outputs

`results/metrics.csv` is the comparison table. `results/error_slices.csv` breaks accuracy down by short tweets, long tweets, negation, laughter, and sarcasm phrases. `results/results.json` stores hyperparameters, influential terms, and misclassified examples. `results/performance.png` and `results/confusion_matrices.png` are the figures.
