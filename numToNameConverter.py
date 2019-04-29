import numpy as np




if __name__ == '__main__':
    trueFrequentNumbers = np.loadtxt(fname="true_frequent_3.txt", delimiter=',', encoding='utf-8')
    stockList = np.loadtxt(fname="data/stocklist.txt", delimiter=',', encoding='utf-8', dtype=str)
    stockList = stockList[0:,0]
    outfile = 'data/trueFrequentTriple'
    outputData = stockList[trueFrequentNumbers.astype(int)]
    np.savetxt(
        fname=outfile,
        X=outputData,
        fmt='%s',
        delimiter=',',
        encoding='utf-8')