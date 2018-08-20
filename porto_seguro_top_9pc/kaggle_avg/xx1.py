

from multiprocessing import *
import warnings
from generic_feat_processing import *

warnings.filterwarnings("ignore")

import xgboost as xgb


def recon(reg):
    integer = int(np.round((40 * reg) ** 2))
    for a in range(32):
        if (integer - a) % 31 == 0:
            A = a
    M = (integer - A) // 31
    return A, M


def transform_df(df):

    df = pd.DataFrame(df)

    dcol = [c for c in df.columns if c not in ['id', 'target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol] == -1).values, axis=1)

    for c in dcol:
        if '_bin' not in c:  # standard arithmetic
            df[c + str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c + str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)

    for c in one_hot:
        if len(one_hot[c]) > 2 and len(one_hot[c]) < 7:
            for val in one_hot[c]:
                df[c + '_oh_' + str(val)] = (df[c].values == val).astype(np.int)
    return df


def multi_transform(df):
    print('Init Shape: ', df.shape)
    p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    p.close();
    p.join()
    print('After Shape: ', df.shape)
    return df

### Gini

def ginic(actual, pred):
    actual = np.asarray(actual)
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n


def gini_normalized(a, p):
    if p.ndim == 2:
        p = p[:, 1]
    return ginic(a, p) / ginic(a, a)


def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score



def run_xgboost_1(chunk_pairs, train, test, y_col, params,  val_repeats = 1):


    dtest = xgb.DMatrix(test, missing=None, weight=None,silent=True,)

    val_preds= np.zeros(train.shape[0])
    test_preds = np.zeros(test.shape[0])

    for i in range(len(chunk_pairs)):

        pair = chunk_pairs[i]
        print " "
        print " ...TRAINING PAIR {} ...".format(i+1)


        assert sorted(pair[0] + pair[1]) == train.index.tolist()

        dtrain = train.loc[pair[0],:]
        dval = train.loc[pair[1],:]

        dtrain = xgb.DMatrix(dtrain.drop([y_col], 1), label=dtrain[y_col], missing=None, silent=True)
        dval = xgb.DMatrix(dval.drop([y_col], 1), label=dval[y_col], missing=None, weight=None, silent=True)

        fit = xgb.train(params, dtrain, feval=gini_xgb, maximize=True,
                        num_boost_round=5000, early_stopping_rounds=100, evals=[(dtrain, "train"), (dval, "val")], verbose_eval=100)

        print " ...BEST SCORE FOR THIS ROUND IS {} ...".format(fit.best_score)

        val_preds[pair[1]] += fit.predict(dval, ntree_limit=fit.best_ntree_limit)
        test_preds += fit.predict(dtest, ntree_limit=fit.best_ntree_limit)

    val_preds = val_preds/val_repeats

    show(" Total val gini is .. {} ".format(gini(train[y_col].values, val_preds)))

    np.save("xx1_val_xgb.npy", val_preds)
    np.save("xx1_test_xgb.npy", (test_preds/len(chunk_pairs)))

    return val_preds, test_preds/len(chunk_pairs)



#### Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


print train.shape
print test.shape

###
y = train['target'].values
testid = test['id'].values

train.drop(['id', 'target'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

### Drop calc
unwanted = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(unwanted, axis=1)
test = test.drop(unwanted, axis=1)

train['ps_reg_A'] = train['ps_reg_03'].apply(lambda x: recon(x)[0])
train['ps_reg_M'] = train['ps_reg_03'].apply(lambda x: recon(x)[1])
train['ps_reg_A'].replace(19, -1, inplace=True)
train['ps_reg_M'].replace(51, -1, inplace=True)

test['ps_reg_A'] = test['ps_reg_03'].apply(lambda x: recon(x)[0])
test['ps_reg_M'] = test['ps_reg_03'].apply(lambda x: recon(x)[1])
test['ps_reg_A'].replace(19, -1, inplace=True)
test['ps_reg_M'].replace(51, -1, inplace=True)

d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
d_skew = train.skew(axis=0)
one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id', 'target']}

train = multi_transform(train)
test = multi_transform(test)

train["target"] = y

print train.shape
print test.shape



params = {'eta': 0.025,
          'max_depth': 4,
          'subsample': 0.9,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'min_child_weight': 100,
          'alpha': 4,
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
          'seed': 99,
          "nthread": 10,
          'silent': True}

chunks = np.load("cv_indx10.npy")
id_pairs = [([y for y in range(10) if y not in x], list(x)) for x in list(itertools.combinations(range(10), 2))]

chunk_pairs = []
for pair in id_pairs:
    chunk_pairs.append(([y for x in pair[0] for y in chunks[x]], [y for x in pair[1] for y in chunks[x]]))


val_preds_xgb1, test_preds_xgb1 = run_xgboost_1(chunk_pairs, train, test, "target", params)
