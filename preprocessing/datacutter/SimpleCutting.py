from CONSTANTS import *


def simple_cut(ids, sequences, shuffle=False):
    idx = list(range(len(ids)))
    if shuffle:
        random.shuffle(idx)
    num_train = math.ceil(0.8 * len(ids))
    train = []
    test = []
    for i in range(num_train + 1):
        train.append((ids[i], sequences[i]))
    for i in range(num_train + 1, len(ids)):
        test.append((ids[i], sequences[i]))
    return train, None, test


def cut_by_82_with_shuffle(instances):
    np.random.shuffle(instances)
    train_split = math.ceil(0.8 * len(instances))
    train = instances[:train_split]
    test = instances[train_split:]
    return train, None, test


def cut_by_73_with_shuffle(instances):
    np.random.shuffle(instances)
    train_split = math.ceil(0.7 * len(instances))
    train = instances[:train_split]
    test = instances[train_split:]
    return train, None, test


def cut_by_613(instances):
    dev_split = math.ceil(0.1 * len(instances))
    train_split = math.ceil(0.6 * len(instances))
    train = instances[:(train_split + dev_split)]
    np.random.shuffle(train)
    dev = train[train_split:]
    train = train[:train_split]
    test = instances[(train_split + dev_split):]
    return train, dev, test


def cut_by_73(instances):
    train_split = int(0.7 * len(instances))
    train = instances[:train_split]
    test = instances[train_split:]
    return train, None, test


def cut_by_613_normal_only(instances, sample_ratio=1):
    '''
    Used by DeepLog mostly. Generate training(with only normal instances), validating, and testing set by 6:1:3. Sampling if given ratio.
    :param instances: Log instances.
    :param sample_ratio: Sampling ratio.
    :return: Training(with normal instances only), validating, and testing set.
    '''
    train, dev, test = cut_by_613(instances)
    new_train = []
    for inst in train:
        if inst.label == 'Normal':
            new_train.append(inst)
    if sample_ratio < 1:
        num_sample = math.ceil(len(new_train) * sample_ratio)
        new_train = random.sample(new_train, num_sample)

    return new_train
