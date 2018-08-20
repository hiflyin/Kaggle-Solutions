import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack
import lightgbm as lgb
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
import sys
sys.path.insert(0, '/wordbatch/')
import wordbatch
from wordbatch.models import FTRL, FM_FTRL
from wordbatch.extractors import WordBag, WordHash

notable = ['bundle', 'new', 'nwt', 'large', 'small', 'sz', 'lot', 'medium', 'xl', 'xs', 'vintage', 'nwot', 'bnwt', ]
remove = ['for', 'and', 'vs', 'of', 'women', 'men', 'ship', 'the', 'on', 'with',
          'in', 'one', 'boys', 'shipping', 'toddler', 'by', 'rm', 'piece', 'me', 'kids', 'fit', 'boy', 'mens', 'super',
          'only', 'all','too', 'color', 'freeship', 'it', 'works', 'like', 'womens', 'to', 'perfect', 'woman', 'youth', 'hello', 'my',
          'from', 'buy','go', 'is', 'at', 'wear', 'toys', 'children']

stopwords = {x: 1 for x in stopwords.words('english')}
stopwords2 = ['abcdef'] + [x for x in stopwords]
non_alphanums = re.compile(u'[^A-Za-z0-9]+')

stemmer = PorterStemmer()


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def stem_tokens(tokens, stemmer):
    stemmed = []

    for item in tokens:
        stemmed_item = item
        try:
            stemmed_item = stemmer.stem(item)
        except Exception:
            print("EXCEPTION WHILE STEM")
            print(stemmed_item)
            pass
        stemmed.append(stemmed_item)

    return stemmed


def fill_missing(df, text_cols, num_cols, bin_cols):
    for col in text_cols:
        df[col].fillna(value="abcdef", inplace=True)
    for col in num_cols:
        df[col].fillna(value=df[col].mean(), inplace=True)
    for col in bin_cols:
        df[col].fillna(value=int(df[col].mean()), inplace=True)


def normalize_text(text):
    return u" ".join([x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] if len(x) > 1 and x not in stopwords])


def filter_norm1(text):
    tokens = [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] if
              len(x) > 1 and x not in stopwords2]
    return " ".join(stem_tokens(tokens, stemmer))


def split_cat(text):
    try:
        return text.split("/")
    except:
        return ["Other", "Other2", "Other3"]


start_time = time.time()

train = pd.read_csv('train.tsv', sep='\t')
test = pd.read_csv('test.tsv', sep='\t')
print('[{}] Finished to load data'.format(time.time() - start_time))
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)

nrow_test = train.shape[0]
outliers = train[(train.price < 1.0)]
train = train.drop(train[(train.price < 1.0)].index)

del outliers['price']
nrow_train = train.shape[0]

y = np.log1p(train["price"])
all_data = pd.concat([train, outliers, test])
submission = test[['test_id']]

all_data['general_cat'], all_data['subcat_1'], all_data['subcat_2'] = zip(
    *all_data['category_name'].apply(lambda x: split_cat(x)))
all_data.drop('category_name', axis=1, inplace=True)

text_cols = ["name", "general_cat", "subcat_1", "subcat_2", "brand_name", "item_description"]
num_cols = ["item_condition_id"]
bin_cols = ["shipping"]
text_seq_cols = ["name", "item_description"]

fill_missing(all_data, text_cols, num_cols, bin_cols)

all_data["all_text"] = all_data["brand_name"].astype(str) + " " + all_data["name"].astype(str) + " " + all_data[
    'item_description']
all_data["name_brand"] = all_data["brand_name"].astype(str) + " " + all_data["name"].astype(str)

wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 1,
                                                              # "hash_ngrams_weights": [1.5, 1.0],
                                                              "hash_size": 2 ** 29,
                                                              "norm": None,
                                                              "tf": 'binary',
                                                              "idf": None, }), procs=8)
wb.dictionary_freeze = True
X_all_text = wb.fit_transform(all_data['all_text'])
del (wb)
X_all_text = X_all_text[:, np.array(np.clip(X_all_text.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
print('[{}] Vectorize `all text` completed.'.format(time.time() - start_time))
print(X_all_text.shape)

wb = wordbatch.WordBatch(filter_norm1, extractor=(WordBag, {"hash_ngrams": 2,
                                                            "hash_ngrams_weights": [1.5, 1.0],
                                                            "hash_size": 2 ** 29,
                                                            "norm": None,
                                                            "tf": 'binary',
                                                            "idf": None, }), procs=8)
wb.dictionary_freeze = True
X_name3 = wb.fit_transform(all_data['name_brand'])
del (wb)
X_name3 = X_name3[:, np.array(np.clip(X_name3.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
print('[{}] Vectorize `name 2-gram` completed.'.format(time.time() - start_time))
print(X_name3.shape)

wb = CountVectorizer()
X_category1 = wb.fit_transform(all_data['general_cat'])
X_category2 = wb.fit_transform(all_data['subcat_1'])
X_category3 = wb.fit_transform(all_data['subcat_2'])
print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

X_dummies = csr_matrix(pd.get_dummies(all_data[['item_condition_id', 'shipping']], sparse=True).values)
print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))

del wb

print(X_dummies.shape, X_category1.shape, X_category2.shape, X_category3.shape, X_name3.shape, X_all_text.shape)
sparse_merge = hstack((X_dummies, X_category1, X_category2, X_category3, X_all_text, X_name3)).tocsr()  # X_brand

print('[{}] Create sparse merge completed'.format(time.time() - start_time))
#
# Remove features with document frequency <=1
print(sparse_merge.shape)
sparse_merge = sparse_merge[:, np.where(sparse_merge.getnnz(axis=0) > 150)[0]]
X = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_test:]
print(sparse_merge.shape)

y = np.log1p(train["price"])

params = {
    'learning_rate': 0.65,
    'application': 'regression',
    'max_depth': 4,
    'num_leaves': 31,
    'verbosity': -1,
    'metric': 'RMSE',
    'data_random_seed': 1,
    'bagging_fraction': 0.8,
    'bagging_freq': 500,
    'feature_fraction': 0.8,
    'nthread': 4,
    'min_data_in_leaf': 100,
    'max_bin': 31
}

d_train = lgb.Dataset(X, label=y)

watchlist = [d_train]
model = lgb.train(params, train_set=d_train, num_boost_round=5500, valid_sets=watchlist, verbose_eval=500)


print('[{}] Finished training model...'.format(time.time() - start_time))

preds1 = model.predict(X_test)

print('[{}] Predict LGB completed.'.format(time.time() - start_time))

model3 = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=sparse_merge.shape[1], alpha_fm=0.01, L2_fm=0.0,
                 init_fm=0.01,
                 D_fm=200, e_noise=0.0001, iters=17, inv_link="identity", threads=4)

model3.fit(X, y)
print('[{}] Train FM_FTRL completed'.format(time.time() - start_time))

preds3 = model3.predict(X_test)

final_pred = 0.39479745 * preds1 + 0.60691396 * preds3

submission['price'] = np.expm1(final_pred)
submission.to_csv("submission_2.csv", index=False)

print('[{}] Finished training models...'.format(time.time() - start_time))
