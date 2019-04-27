import numpy as np


if __name__ == "__main__":

    infile = 'data/baskets2017.txt'
    outfile = 'data/basketsenum2017.txt'

    L = np.loadtxt('data/stocklist.txt',delimiter=',',dtype=str)
    L = L[:,0]
    # D = np.zeros(shape=(8582,505),dtype=int)
    D = np.zeros(shape=(1526,505),dtype=int)

    @np.vectorize
    def enum(s):
        t = s.upper()
        t = t.strip('\'')
        index = np.argwhere(L == t)
        if (len(index) != 0):
            return index[0][0]
        else:
            return -1

    with open(infile) as f:

        i = 0

        for line in f.readlines():
            A = line.split(', ')
            A = np.array(A,dtype=str)
            indices = enum(A)
            indices = indices[np.where(indices != -1)]
            d = np.zeros(505)
            d[indices] = 1
            D[i] = d
            i += 1
            if (i % 100 == 0):
                print(i)

    print('enumeration complete')

    D = D.astype(int)

    np.savetxt(fname=outfile, 
                X=D,
                fmt='%i',
                delimiter= ',', 
                encoding = 'utf-8')