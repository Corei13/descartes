import os, sys, csv, numpy, zipfile

datadir = '../data'

trainfile, testfile, outfile = sys.argv[1:4] if len(sys.argv) == 4 else ('training.zip', 'test.zip', 'data.new.npz')
filter_mass, filter_jet_zero, filter_jet_one = [1], [4, 5, 6, 12, 22, 23, 24, 25, 26, 27], [4, 5, 6, 12, 25, 26, 27]
trainX, trainW, trainY, testX = None, None, None, None

fileType = lambda path: path.split('.')[-1]
getZipFile = lambda path: zipfile.ZipFile(path)
getCsvFile = lambda zfile: zfile.open(zfile.namelist()[0])
getCsvReader = lambda path: csv.reader(open(path) if fileType(path) == 'csv' else getCsvFile(getZipFile(path)))

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
        # print("Stat: hist(jet_num): %s, hist(y): %s, total weight: %s" % (map(sum, [jet_num == i for i in [0, 1, 2, 3]]), [sum(y == 1), sum(y == 0)], sum(w)))
        return X, y, w
    else:
        # print("Stat: hist(jet_num): %s" % (map(sum, [jet_num == i for i in [0, 1, 2, 3]])))
        return X

print ('processing data files %s and %s ...' % (trainfile, testfile))
(trainX, trainY, trainW), testX = getData(trainfile, dtype = 'training'), getData(testfile, dtype = 'test')
print ('saving processed data to %s ...' % outfile)
numpy.savez_compressed(os.path.join(datadir, outfile),
    X = numpy.concatenate([trainX, testX]), w = trainW, y = trainY, filters = (filter_mass, filter_jet_zero, filter_jet_one))
