from ctypes import *
import numpy as np

libmodel = np.ctypeslib.load_library('libpolicy_export_toy.so', './tensorflow/bazel-bin/')
libmodel.run.argtypes = [
    np.ctypeslib.ndpointer(np.int32, ndim=1, shape=(1,), flags=('c', 'a')),
    np.ctypeslib.ndpointer(np.int32, ndim=1, shape=(1,), flags=('c', 'a')),
    np.ctypeslib.ndpointer(np.int32, ndim=1, shape=(1,), flags=('c', 'a', 'w'))]

libmodel.CreateNetwork()

def predict(x, y):
    x = np.require(x, np.int32, ('c', 'a'))
    y = np.require(y, np.int32, ('c', 'a'))
    z = np.require(np.zeros((1)), np.int32, ('c', 'a', 'w'))
    libmodel.run(x, y, z)
    return z

a = np.array([4])
b = np.array([6])
c = predict(a,b)
print('{} + {} = {}'.format(a[0], b[0], c[0]))

libmodel.DeleteNetwork()
