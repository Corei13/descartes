import os

from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse


@no_debug_mode
def train_yaml(yaml_file):

    train = yaml_parse.load(yaml_file)
    train.main_loop()


def train_layer1(yaml_file_path, save_path):

    yaml = open("dae/layer-01.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'train_stop': 800000,
                    'batch_size': 50,
                    'monitoring_batches': 1,
                    'nhid': 1000,
                    'max_epochs': 50}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)


def train_layer2(yaml_file_path, save_path):

    yaml = open("dae/layer-02.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'train_stop': 800000,
                    'batch_size': 50,
                    'monitoring_batches': 1,
                    'nvis': 1000,
                    'nhid': 1000,
                    'max_epochs': 50}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)


def train_layer3(yaml_file_path, save_path):

    yaml = open("dae/layer-03.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'train_stop': 800000,
                    'batch_size': 50,
                    'monitoring_batches': 1,
                    'nvis': 1000,
                    'nhid': 1000,
                    'max_epochs': 50}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)


def train_mlp(yaml_file_path, save_path):

    yaml = open("dae/mlp.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'train_stop': 200000,
                    'valid_stop': 250000,
                    'batch_size': 50,
                    'max_epochs': 100}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)


def test_sda():

    yaml_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                  '..'))
    save_path = os.path.dirname(os.path.realpath(__file__))

    train_layer1(yaml_file_path, save_path)
    train_layer2(yaml_file_path, save_path)
    train_layer3(yaml_file_path, save_path)
    train_mlp(yaml_file_path, save_path)

if __name__ == '__main__':
    test_sda()
