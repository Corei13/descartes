import os, sys, csv, numpy, zipfile

datadir = '../data'
missing_value = -999.0
global_N = (411000.0, 692.0)

trainfile, testfile, outfile = sys.argv[1:4] if len(sys.argv) == 4 else ('training.zip', 'test.zip', 'data.npz')
filter_mass, filter_jet_zero, filter_jet_one = [0], [4, 5, 6, 12, 22, 23, 24, 25, 26, 27], [4, 5, 6, 12, 25, 26, 27]
trainX, trainW, trainY, testX = None, None, None, None

fileType = lambda path: path.split('.')[-1]
getZipFile = lambda path: zipfile.ZipFile(path)
getCsvFile = lambda zfile: zfile.open(zfile.namelist()[0])
getCsvReader = lambda path: csv.reader(open(path) if fileType(path) == 'csv' else getCsvFile(getZipFile(path)))

def normalize(X):
    def func(x):
        norm = lambda a: 0.5 + (a - a.min()) / max(2.0 * a.ptp(), 1e-6)
        missing = (x == missing_value)
        y = numpy.zeros(x.shape)
        y[-missing] = norm(x[-missing])
        y[ missing] = 0.0
        return y
    return numpy.apply_along_axis(func, 0, X)

def getData(datafile, dtype):
    path = os.path.join(datadir, datafile)
    cr = getCsvReader(path)
    header = cr.next()
    data = numpy.asarray([ row for row in cr ])
    jet_num = numpy.asarray(data[:,23], dtype = int)
    ext_jet_num = numpy.concatenate( [ [ numpy.asarray(jet_num >= i, dtype = int) ] for i in [1, 2, 3] ] ).T
    X = numpy.asarray(numpy.concatenate([data[:,1:23], data[:,24:31], ext_jet_num], axis = 1), dtype = float)
    if dtype == 'training':
        w = numpy.asarray(data[:,31], dtype = float)
        y = numpy.asarray(data[:,32] == 's', dtype = int)
        N = (sum(w[y == 0]), sum(w[y == 1]))
        
        w[y == 0] *= global_N[0] / sum(w[y == 0])
        w[y == 1] *= global_N[1] / sum(w[y == 1])

        print("Stat: hist(jet_num): %s, hist(y): %s, total weight: %s" % (map(sum, [jet_num == i for i in [0, 1, 2, 3]]), [sum(y == 1), sum(y == 0)], [sum(w[y == 0]), sum(w[y == 1])]))
        return X, y, w
    else:
        # print("Stat: hist(jet_num): %s" % (map(sum, [jet_num == i for i in [0, 1, 2, 3]])))
        return X

print ('processing data files %s and %s ...' % (trainfile, testfile))
(trainX, trainY, trainW), testX = getData(trainfile, dtype = 'training'), getData(testfile, dtype = 'test')
print ('saving processed data to %s ...' % outfile)
numpy.savez_compressed(os.path.join(datadir, outfile),
    X = normalize(numpy.concatenate([trainX, testX])), w = trainW, y = trainY.reshape((trainY.shape[0], 1)), filters = (filter_mass, filter_jet_zero, filter_jet_one))
