from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy

class HiggsDataset(DenseDesignMatrix):
    def __init__(self, file='../data/data.npz', which_set=None, filters=None, start=None, stop=None):
        assert file is not None, ("File name is needed!")
        
        data = numpy.load(file)
        X, y, w, filters = data['X'], data['y'], data['w'], data['filters']
        
        data_set_size = X.shape[0]
        train_set_size = y.shape[0]
        test_set_size = data_set_size - train_set_size

        n_features = X.shape[1]

        if which_set == 'train':
            start, stop = 0 if start is None else start, train_set_size if stop is None else stop
            assert start < stop and stop <= train_set_size

            super(HiggsDataset, self).__init__(X=X[start:stop], y=y[start:stop], y_labels=2)
            self.w = w[start:stop]
        elif which_set == 'test':
            start, stop = 0 if start is None else start, test_set_size if stop is None else stop
            assert start < stop and stop <= test_set_size

            super(HiggsDataset, self).__init__(X=X[train_set_size+start:train_set_size+stop])
        else:
            start, stop = 0 if start is None else start, data_set_size if stop is None else stop
            assert start < stop and stop <= data_set_size

            super(HiggsDataset, self).__init__(X=X[start:stop])
