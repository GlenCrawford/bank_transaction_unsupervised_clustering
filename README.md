# Classifying my bank transactions with unsupervised clustering

All of my machine learning projects so far have been involved "supervised learning", where a model is trained on data that is labelled with the correct result, so that it can learn how to make an accurate prediction for unlabelled data. However, we don't always have the luxury of labelled datasets; sometimes all we have is a raw dataset that we have to divine some meaning out of from scratch.

Thus, this is project to try out some methods to classify data without having labels in advance. The mobile app for my bank automatically classifies each transaction as it happens into broad categories: "Groceries", "Utilities", "Travel", etc. I looked and found that my bank allows me to export my transactions as CSV files for the current and last financial years. The categories aren't in there, so I figured I would see if I could replicate them somewhat.

A couple of notes. First, I don't expect that unsupervised clustering is the way that my bank does it. They probably have customers that have transacted with nearly every retailer in the country, so they probably already have a good database built up of which category to apply to transactions from each merchant, for example, that a debit card transaction from a Woolworths store is for groceries, or that a credit card transaction from the Qantas website is for travel. This model starts from scratch with none of that inherant knowledge, so I never really expected it to be as good as the bank's. Realistically, this is probably better suited to being a supervised model, where I go through and manually label the training data myself. But then I would need to come up with another excuse to try out clustering algorithms ;)

Second, the data in the CSV files is disappointinly minimal. Here is what a few (anonymized and edited) lines look like:

| Date       | Amount   | Merchant                                                              | Balance After |
|:----------:|:--------:|:---------------------------------------------------------------------:|:-------------:|
| 30/02/2020 | -23.64   | KFC SWANSTON401          MELBOURNE    AU Card xx3819                  | +3749.44      |
| 07/11/2019 | +3000.00 | Transfer from xx1831                                                  | +5021.78      |
| 21/03/2019 | -199.50  | TRANSPORT FOR NSW CHIPPENDALE  AUS Card xx3819 Value Date: 21/03/2019 | +2587.70      |

(The column names aren't in the files, only data rows.)

As you can see, there's not as much detail as there could be. The bank undoubtedly also has other details, such as the time of day. Two of the columns aren't even useful for determining the category of the transaction (Date and Balance After), so there wasn't much to work with.

The results of the model aren't spectacular; I wasn't really expecting them to be, given how low dimension the data is. With more details available, the model would likely be more accurate. For example, I would have liked to use the hour of day, perhaps bucketized into groups three or four hours wide, since I expect that most of my grocery transactions happen in the afternoons, yet most of my entertainment transactions happen in the evenings. Regardless, the model does a pretty good job at identifying some well-grouped clusters of transactions, such as groceries, rent payments, internal transfers between accounts, utility bills, even airline ticket purchases.

## How it works

Traditionally, unsupervised machine learning is done via clustering, where each example is plotted on an n-dimensional space (n being the number of features) and the algorithm attemps to group them into clusters based on the similarity of their features by:

* Plotting each example in an n-dimensional space.
* Randomly placing `k` number of "centroids" (which will come to represent the center of a cluster) on the graph.
* Assigning each example to the nearest centroid by calculating each example's position on the graph's Euclidean distance from each centroid and assigning them to the closest one.
* Moving each centroid to the center of its cluster of assigned examples.
* Repeating steps 3 and 4 until no examples are reassigned to a different cluster in an interation.

This is usually done with the k-means algorithm, however, that only works when all features are numeric, and we have features, such as Merchant, that are categorical (ie, there is a finite list of all possible values). Conversely, the k-modes algorithm is designed only for clustering categorical features, but the Amount feature is obviously numeric. So I needed to use an algorithm that can handle a mix of both: k-prototypes, with combines k-means and k-modes.

Unfortunately, the k-modes and k-prototypes algorithms are not supported by scikit-learn. However, there is a [kmodes library](https://pypi.org/project/kmodes/) that implements them. The APIs are modelled after that of the clustering algorithms that scikit-learn does support.

## Data preprocessing

### Normalizing the Merchant

As you can see in the samples above, the "Merchant" column is very messy. It's not even just the merchant; it appears to also contain reference numbers, particulars, dates, etc. It also includes transactions without a merchant, such as transfers between accounts. I decided to normalize it in two ways: remove all but the merchant name, and make sure there aren't variations of the same merchant, e.g. "HNGY JCK SYD INT ARPT SYDNEY INTERN", "Hungry Jack?s Melbourn   Melbourne    AU" and "HJ MICHAELS CORNER MELBOURNE  AUS Card xx3819 Value Date: 19/11/2019" should all just be "Hungry Jack's":

```python
MERCHANT_NORMALIZATION_MAPPING_EXPRESSIONS = {
  r'.*KFC.*': 'KFC',
  r'.*Transfer.*': 'Transfer',
  r'.*TRANSPORT FOR NSW.*': 'Transport for New South Wales'
}
```

After applying this mapping, every transaction has a clean "Merchant" value; the same value as every other transaction with the same merchant, and stripped of all the dates, card numbers, etc.

### Adding a Transaction Type feature

Given that I only had two usable features, I decided to manually add a third one: Transaction Type. I created a mapping that looks like the following, and populated it from memory and a dash of common sense:

```python
MERCHANT_TRANSACTION_TYPE_MAPPINGS = {
  'KFC': DEBIT_CARD_TRANSACTION_TYPE,
  'Qantas': CREDIT_CARD_ONLINE_TRANSACTION_TYPE,
  'Woolworths': DEBIT_CARD_TRANSACTION_TYPE
}
```

For example, I know that my KFC transactions are made in-store by debit card, and that my Qantas transactions are online by credit card.

Then I bulk-populated the transactions in the dataset with the type of each transaction, matched by the post-normalized merchant name. This gave me a third usable feature to use in training. That's three dimensions to the dataset, before category encoding.

### Category encoding

The above transactions were done with Pandas, as well as one-hot encoding of the two categorical features: Merchant and Transaction Type.

## Requirements

Python version: Developed with 3.8.1

See dependencies.txt for packages and versions (and below to install).

## Setup

Clone the Git repository.

Install the dependencies:

```bash
pip install -r dependencies.txt
```

Export your own bank transactions as CSV and place them in the `data` directory.

### Mappings

The `data/merchant_normalization_mapping_expressions.py` file is not included in the repo, partly to protect my privacy, and partly to hide my shame at how often I go to KFC.

Create the file yourself, and fill it in with the following template:

```python
MERCHANT_NORMALIZATION_MAPPING_EXPRESSIONS = {
  r'.*Expression to match the merchant name.*': 'Merchant name',
  ...
}

DEBIT_CARD_TRANSACTION_TYPE = 'Debit card'
...

MERCHANT_TRANSACTION_TYPE_MAPPINGS = {
  'Merchant name': DEBIT_CARD_TRANSACTION_TYPE,
  ...
}
```

## Run

```bash
python -W ignore main.py
```

The script will conclude by printing out each transaction grouped by cluster. The clusters don't have names, just an integer starting at zero.
