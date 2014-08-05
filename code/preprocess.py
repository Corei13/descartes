import os, sys, csv, numpy, zipfile

datadir = '../data'
trainfile, testfile, outfile = 'training.zip', 'test.zip', 'data.npz'

filter_mass, filter_jet_zero, filter_jet_one = [1], [4, 5, 6, 12, 22, 23, 24, 25, 26, 27], [4, 5, 6, 12, 25, 26, 27]
trainX, trainW, trainY, testX = None, None, None, None

with zipfile.ZipFile(os.path.join(datadir, trainfile)) as zfile:
    csvfile = zfile.open(zfile.namelist()[0])
    cr = csv.reader(csvfile)
    header = cr.next()
    data = numpy.asarray([ row for row in cr ])
    jet_num = numpy.asarray(data[:,23], dtype = int)
    ext_jet_num = numpy.concatenate( [ [ numpy.asarray(jet_num >= i, dtype = int) ] for i in [1, 2, 3] ] ).T
    trainX = numpy.asarray(numpy.concatenate([data[:,1:23], data[:,24:31], ext_jet_num], axis = 1), dtype = float)
    trainW = numpy.asarray(data[:,31], dtype = float)
    trainY = numpy.asarray(data[:,32] == 's', dtype = int)
    print("Stat: hist(jet_num): %s, hist(y): %s, total weight: %s" % (map(sum, [jet_num == i for i in [0, 1, 2, 3]]), [sum(trainY == 1), sum(trainY == 0)], sum(trainW)))


with zipfile.ZipFile(os.path.join(datadir, testfile)) as zfile:
    csvfile = zfile.open(zfile.namelist()[0])
    cr = csv.reader(csvfile)
    header = cr.next()
    data = numpy.asarray([ row for row in cr ])
    jet_num = numpy.asarray(data[:,23], dtype = int)
    ext_jet_num = numpy.concatenate( [ [ numpy.asarray(jet_num >= i, dtype = int) ] for i in [1, 2, 3] ] ).T
    testX = numpy.asarray(numpy.concatenate([data[:,1:23], data[:,24:31], ext_jet_num], axis = 1), dtype = float)
    print("Stat: hist(jet_num): %s" % (map(sum, [jet_num == i for i in [0, 1, 2, 3]])))

numpy.savez_compressed(os.path.join(datadir, outfile),
    X = numpy.concatenate([trainX, testX]), W = trainW, y = trainY, filters = (filter_mass, filter_jet_zero, filter_jet_one))
