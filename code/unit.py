from higgsdataset import HiggsDataset

from pylearn2.termination_criteria import EpochCounter
from pylearn2.testing.skip import skip_if_no_data
from pylearn2.config import yaml_parse

with open('mlp.test.yaml', 'r') as f:
    train = f.read()
hyper_params = {'train_stop': 50,
                'valid_start':51,
                'valid_stop': 100,
                'dim_h0': 5,
                # 'n_feat': ,
                'max_epochs': 1,
                'save_path': '../junk'}
train = train % (hyper_params)
train = yaml_parse.load(train)
train.main_loop()
