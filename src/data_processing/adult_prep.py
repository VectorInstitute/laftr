import numpy as np
import sys

# make noNULL file with: grep -v NULL rawdata_mkmk01.csv | cut -f1,3,4,6- -d, > rawdata_mkmk01_noNULL.csv
EPS = 1e-8
#### LOOK AT THIS FUNCTION!!!! GETTING STD = 0
def bucket(x, buckets):
    x = float(x)
    n = len(buckets)
    label = n
    for i in range(len(buckets)):
        if x <= buckets[i]:
            label = i
            break
    template = [0. for j in range(n + 1)]
    template[label] = 1.
    return template

def onehot(x, choices):
    if not x in choices:
        print('could not find "{}" in choices'.format(x))
        print(choices)
        raise Exception()
    label = choices.index(x)
    template = [0. for j in range(len(choices))]
    template[label] = 1.
    return template

def continuous(x):
    return [float(x)]


def parse_row(row, headers, headers_use):
    new_row_dict = {}
    for i in range(len(row)):
        x = row[i]
        hdr = headers[i]
        new_row_dict[hdr] = fns[hdr](x)
    sens_att = new_row_dict[sensitive]
    label = new_row_dict[target]
    new_row = []
    for h in headers_use:
        new_row = new_row + new_row_dict[h]
    return new_row, label, sens_att

def whiten(X, mn, std):
    mntile = np.tile(mn, (X.shape[0], 1))
    stdtile = np.maximum(np.tile(std, (X.shape[0], 1)), EPS)
    X = X - mntile
    X = np.divide(X, stdtile)
    return X


if __name__ == '__main__':
    f_in_tr = '/ais/gobi4/madras/adult/adult.data'
    f_in_te = '/ais/gobi4/madras/adult/adult.test'

    f_out_np = '/ais/gobi4/madras/adult/adult.npz'
    hd_file = '/ais/gobi4/madras/adult/adult.headers'
    f_out_csv = '/ais/gobi4/madras/adult/adult.csv'

    header_list = open(hd_file, 'w')

    REMOVE_MISSING = True
    MISSING_TOKEN = '?'

    headers = 'age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income'.split(',')
    headers_use = 'age,workclass,education,education-num,marital-status,occupation,relationship,race,capital-gain,capital-loss,hours-per-week,native-country'.split(',')
    target = 'income'
    sensitive = 'sex'

    options = {
        'age': 'buckets',
        'workclass': 'Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked',
        'fnlwgt': 'continuous',
        'education': 'Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool',
        'education-num': 'continuous',
        'marital-status': 'Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse',
        'occupation': 'Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces',
        'relationship': 'Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried',
        'race': 'White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black',
        'sex': 'Female, Male',
        'capital-gain': 'continuous',
        'capital-loss': 'continuous',
        'hours-per-week': 'continuous',
        'native-country': 'United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands',
        'income': ' <=50K,>50K'
    }

    buckets = {'age': [18, 25, 30, 35, 40 ,45, 50, 55, 60, 65]}

    options = {k: [s.strip() for s in sorted(options[k].split(','))] for k in options}

    fns = {
        'age': lambda x: bucket(x, buckets['age']),
        'workclass': lambda x: onehot(x, options['workclass']),
        'fnlwgt': lambda x: continuous(x),
        'education': lambda x: onehot(x, options['education']),
        'education-num': lambda x: continuous(x),
        'marital-status': lambda x: onehot(x, options['marital-status']),
        'occupation': lambda x: onehot(x, options['occupation']),
        'relationship': lambda x: onehot(x, options['relationship']),
        'race': lambda x: onehot(x, options['race']),
        'sex': lambda x: onehot(x, options['sex']),
        'capital-gain': lambda x: continuous(x),
        'capital-loss': lambda x: continuous(x),
        'hours-per-week': lambda x: continuous(x),
        'native-country': lambda x: onehot(x, options['native-country']),
        'income': lambda x: onehot(x.strip('.'), options['income']),
    }

    D = {}
    for f, phase in [(f_in_tr, 'training'), (f_in_te, 'test')]:
        dat = [s.strip().split(',') for s in open(f, 'r').readlines()]

        X = []
        Y = []
        A = []
        print(phase)

        for r in dat:
            row = [s.strip() for s in r]
            if MISSING_TOKEN in row and REMOVE_MISSING:
                continue
            if row in ([''], ['|1x3 Cross validator']):
                continue
            newrow, label, sens_att = parse_row(row, headers, headers_use)
            X.append(newrow)
            Y.append(label)
            A.append(sens_att)

        npX = np.array(X)
        npY = np.array(Y)
        npA = np.array(A)
        npA = np.expand_dims(npA[:,1], 1)

        D[phase] = {}
        D[phase]['X'] = npX
        D[phase]['Y'] = npY
        D[phase]['A'] = npA

        print(npX.shape)
        print(npY.shape)
        print(npA.shape)

    #should do normalization and centring
    mn = np.mean(D['training']['X'], axis=0)
    std = np.std(D['training']['X'], axis=0)
    print(mn, std)
    D['training']['X'] = whiten(D['training']['X'], mn, std)
    D['test']['X'] = whiten(D['test']['X'], mn, std)

    #should write headers file
    f = open(hd_file, 'w')
    i = 0
    for h in headers_use:
        if options[h] == 'continuous':
            f.write('{:d},{}\n'.format(i, h))
            i += 1
        elif options[h][0] == 'buckets':
            for b in buckets[h]:
                colname = '{}_{:d}'.format(h, b)
                f.write('{:d},{}\n'.format(i, colname))
                i += 1
        else:
            for opt in options[h]:
                colname = '{}_{}'.format(h, opt)
                f.write('{:d},{}\n'.format(i, colname))
                i += 1

    n = D['training']['X'].shape[0]
    shuf = np.random.permutation(n)
    valid_pct = 0.2
    valid_ct = int(n * valid_pct)
    valid_inds = shuf[:valid_ct]
    train_inds = shuf[valid_ct:]

    np.savez(f_out_np, x_train=D['training']['X'], x_test=D['test']['X'],
                y_train=D['training']['Y'], y_test=D['test']['Y'],
                attr_train=D['training']['A'], attr_test=D['test']['A'],
             train_inds=train_inds, valid_inds=valid_inds)





