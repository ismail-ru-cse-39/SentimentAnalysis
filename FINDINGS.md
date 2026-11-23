# Sentiment analysis findings

These notes match the Practical Machine Learning report sections. The image-classification half is still separate. In the IEEE paper this material has to be shortened so both tasks fit in four double-column pages.

## Experimental setup

The corpus is the supplied Twitter collection. Each line is one tweet. The label is the filename, not a column in the file.

| Split | Positive | Negative |
|---|---:|---:|
| Training pool | 399,000 | 399,000 |
| Test (`ttestPos.en`, `ttestNeg.en`) | 1,000 | 998 |

The training pool is balanced, so a stratified sample of 50,000 tweets per class was drawn with seed 42. That sample was split 90/10 into 90,000 training tweets and 10,000 validation tweets. The test files were not sampled and were not used to fit any vectorizer, embedding, or classifier.

Cleaning, applied to every tweet before vectorization:

- lowercase text and strip a surrounding pair of quotation marks
- remove URLs and @mentions
- keep the word inside a hashtag and drop the `#`
- expand contractions such as `n't` so that negation survives punctuation removal
- remove digits and punctuation
- leave stopwords in, because words such as `not` and `no` carry the sentiment

After cleaning, the training sample averages 13.0 tokens (median 12, 95th percentile 26). 206 sampled tweets and 6 test tweets are empty, usually because the original text was only a URL or a mention. Empty tweets become zero vectors.

Vectorizer settings, shared by the count, TF, IDF, and TF-IDF pipelines: unigrams and bigrams, `min_df=2`, at most 20,000 features. The classifier is L2 logistic regression (`liblinear`). The regularisation strength `C` is chosen from {0.1, 1, 4} by validation macro F1, then the vectorizer and classifier are refit on the full 100,000-tweet sample before the test evaluation.

Word2Vec is skip-gram, 100 dimensions, window 5, minimum count 2, 5 epochs, trained only on the training text. Each tweet is the mean of its in-vocabulary word vectors, then standardised. Nine test tweets had no in-vocabulary tokens.

## Methodology

All learned pipelines use the same classifier, so the comparison is between representations.

- **Lexicon baseline.** A fixed list of positive and negative words. The tweet is positive when the positive count exceeds the negative count. Ties, which are tweets with none of the listed words, are predicted positive. This is a reference point, not one of the required representations.
- **Bag-of-words.** Raw unigram and bigram counts.
- **TF.** The same n-grams, replaced by their L2-normalised term frequency. IDF is turned off.
- **IDF.** Each n-gram is recorded as present or absent, then multiplied by smoothed inverse document frequency. This isolates the IDF weight from how often a word is repeated inside one short tweet.
- **TF-IDF.** Sublinear term frequency multiplied by smoothed IDF, then L2-normalised.
- **Dense embeddings.** Mean-pooled skip-gram Word2Vec vectors.

On the TF-IDF model the lowest IDF terms are function words (`to`, `the`, `my`, `and`, `it`, `you`) together with `not`. IDF therefore down-weights words that appear in both classes. The highest IDF terms are rare doubled tokens and slang; L2 regularisation stops those rare terms from dominating the decision.

## Results

Test set, 1,998 tweets. Macro F1 is the selection metric because it treats both classes equally.

| Pipeline | Accuracy | Macro F1 | Positive F1 | Negative F1 | Chosen C |
|---|---:|---:|---:|---:|---:|
| Lexicon | 0.598 | 0.531 | 0.708 | 0.354 | — |
| BoW counts | 0.799 | 0.799 | 0.803 | 0.795 | 0.1 |
| TF | 0.800 | 0.800 | 0.801 | 0.800 | 4.0 |
| IDF | 0.806 | 0.806 | 0.806 | 0.805 | 1.0 |
| TF-IDF | 0.801 | 0.801 | 0.802 | 0.800 | 1.0 |
| Word2Vec mean | 0.746 | 0.746 | 0.743 | 0.748 | 0.1 |

The lexicon looks better on accuracy than on macro F1 because 1,215 of 1,998 test tweets are ties and are all predicted positive. Negative recall is only 0.220. Every learned model beats this baseline by a wide margin.

The four sparse pipelines sit within one percentage point of each other. IDF is numerically highest. With 1,998 test tweets, a binomial standard error around 80% accuracy is about 0.9 points, so a 95% interval is roughly ±1.8 points. The gaps among BoW, TF, IDF, and TF-IDF are inside that interval. The useful result is that, on short tweets, changing the weighting from raw counts to TF or TF-IDF does not change the decision much. Repeating a word inside one tweet adds little once bigrams are present.

Word2Vec is about six points lower. That gap is larger than the sampling noise. Averaging word vectors throws away order, so `not good` is pulled toward both `not` and `good`. It also gives a vector to topical tokens such as `followfriday` and `retweet`, which co-occur with positive tweets without being sentiment words.

The learned sparse weights support that reading. Positive bigrams include `not wait`, `not bad`, and `cant wait`. Negative unigrams include `sad`, `miss`, `sick`, `sucks`, and `hate`. The bigram model can treat `not bad` as positive while still treating `not` on its own as negative. A mean embedding cannot make that distinction.

Figures: `results/performance.png` and `results/confusion_matrices.png`. For IDF the test confusion matrix is 802 true negatives, 196 false positives, 192 false negatives, and 808 true positives.

## Error analysis

Errors below are from the IDF model unless a comparison says otherwise.

| Slice of the test set | Tweets | IDF accuracy | Word2Vec accuracy |
|---|---:|---:|---:|
| 3 tokens or fewer | 137 | 0.847 | 0.774 |
| 25 tokens or more | 150 | 0.773 | 0.740 |
| Contains a negation word | 506 | 0.794 | 0.719 |
| Contains laughter (`lol`, `haha`, …) | 112 | 0.795 | 0.741 |

Short tweets are easier because they usually contain one clear cue. Longer tweets often mix a positive event with a complaint, and the model has to pick one label. Negation is where the embedding loses the most: its accuracy falls to 0.719, against 0.794 for IDF. Only two test tweets matched the hand-written sarcasm phrases, so sarcasm is not measurable as its own slice here. It shows up instead inside mixed tweets.

High-confidence mistakes are mostly mixed or context-dependent, not random:

- False positives include tweets that say `Happy Sunday` or `Happy Mother's Day` and then mention school, pain, or being unable to sleep. The positive phrase dominates the weight vector.
- False negatives include `11:11 make a wish`, `I miss you`, and `I hate the heat … Go Norway`. `wish` and `miss` are strong negative weights, including when the tweet is affectionate or hopeful. Several of these gold labels also look noisy: a tweet whose visible text is clearly positive is filed in the negative test file. That label noise is a ceiling on accuracy.

TF-IDF is right and Word2Vec is wrong on 213 test tweets. The reverse happens on 102. The tweets Word2Vec misses include explicit negation and contrast (`Can't wait`, `so sad I'm not there`, `good looking guy … just got killed`). BoW and TF-IDF disagree on only 65 test tweets (30 one way, 35 the other), which matches the near-tie in the table.

## What to carry into the report

State the sample size and the untouched test files. Report the table above, and say that BoW, TF, IDF, and TF-IDF are practically tied while mean-pooled embeddings are worse because they discard order and negation. Use `not bad` versus `not`, and one mixed tweet, as the error-analysis examples. Re-run with `PYTHONHASHSEED=42 python run_sentiment.py` to regenerate every number.
