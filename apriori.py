from itertools import chain
from itertools import combinations
import numpy as np
import scipy.sparse as sp
from scipy.special import comb


# takes a tuple T and returns a hashed index
def pcy_hash(n,T):

    @np.vectorize
    def hash_items(item):
        return hash(item)

    # TODO perhaps write a better hash
    indices = hash_items(T)
    index = 0
    for i in range(len(T)):
        index += indices[i]**(i + 4)
    index = index % n
    return index


# returns all possible combinations 
# of size n within array A 
# made up of frequent combos in S
def itemcombos(A,n,S):

    # filter combos that are not made from 
    # truly frequent tuples
    def filter_tuple(T):
        prev = chain.from_iterable(combinations(T,n-1))
        prev = np.fromiter(prev,dtype=int)
        prev = prev.reshape((len(prev) // n-1),n-1) 
        prev = np.isin(T,S)
        if (np.sum(prev) == n):
            return T

    combos = chain.from_iterable(combinations(A,n))
    combos = np.fromiter(combos,dtype=int)
    combos = combos.reshape((len(combos) // n),n)
    if n > 2:
        combos = np.apply_along_axis(filter_tuple,1,combos)
    return combos


# Associative rule mining for dataset D as a 
# sparse marix where s is support for k tuples
def pcy(D,s,k):
    rows = D.shape[0] # number of baskets
    cols = D.shape[1] # number of items  
    bitmap = None
    bucket_size = comb(cols,2) // 2
    true_frequent = np.arange(1,cols+1)

    # increments buckets given
    # an array of hash values
    @np.vectorize
    def increment_buckets(index):
        buckets[index] += 1

    # check if tuple T hashes to frequent bucket
    def hashto(T,indices,n):
        A = np.array([pcy_hash(n,T)])
        if np.isin(A,indices)[0]:
            return T
        

    # loop through finding frequent 
    # k-tuples of support s
    for i in range(1, (2*k)):
        if (i % 2 == 1):
            j = (i // 2) + 2
            buckets = np.zeros(
                shape=(comb(len(true_frequent),j,exact=True) // 2), dtype=int)
            for r in range(rows):
                print('count:%i' % (r))

                basket = D[r,:]
                basket = basket.toarray()
                basket = np.argwhere(basket == 1)[:,1]

                # no possible combo can be frequent, so skip
                if len(basket) <= j:
                    continue

                # generate canidate tuples
                canidates = itemcombos(basket,j,true_frequent)

                # hash each canidate tuple in canidates and increment bucket
                hash_tuple = lambda x: pcy_hash(len(buckets), x)
                hash_indices = np.apply_along_axis(hash_tuple,1,canidates)
                increment_buckets(hash_indices)
            print(buckets.shape)
            np.savetxt(
                fname='buckets1.txt',
                X=buckets,
                fmt='%i',
                delimiter=',',
                encoding='utf-8')
            bitmap = np.where(buckets > s,1,0)
            np.savetxt(
                fname='bitmap1.txt',
                X=bitmap,
                fmt='%i',
                delimiter=',',
                encoding='utf-8')
            print(bitmap.shape)
        else:
            for r in range(rows):
                print('count:%i' % (r))
                basket = D[r,:]
                basket = basket.toarray()
                basket = np.argwhere(basket == 1)[:,1]

                if len(basket) <= j:
                    continue
                
                # generate canidate tuples
                canidates = itemcombos(basket,j,true_frequent)
                np.savetxt(
                    fname='canidates.txt',
                    X=canidates,
                    fmt='%i',
                    delimiter=',',
                    encoding='utf-8')
                # hash each canidate tuple in canidates and increment bucket
                hash_tuple = lambda x: pcy_hash(len(buckets), x)
                
                hash_indices = np.apply_along_axis(hash_tuple,1,canidates)
                np.savetxt(
                    fname='hash_indices.txt',
                    X=hash_indices,
                    fmt='%i',
                    delimiter=',',
                    encoding='utf-8')

                # we neeed the hash index where bitmap[index] ==  1
                freq_buckets = np.argwhere(bitmap == 1)
                hash_indices = np.intersect1d(freq_buckets,hash_indices)                

                np.savetxt(
                    fname='freq_hash_indices.txt',
                    X=hash_indices,
                    fmt='%i',
                    delimiter=',',
                    encoding='utf-8')
                
                check_hash_tuple = lambda x: hashto(x,hash_indices,len(buckets))
                true_canidates = np.apply_along_axis(check_hash_tuple,1,canidates)
                np.savetxt(
                    fname='true_canidates.txt',
                    X=true_canidates,
                    fmt='%i',
                    delimiter=',',
                    encoding='utf-8')
                break






                


if __name__ == '__main__':
    # set support to 860 such that stocks are similar to
    # are 70 percent of the days in the last five years
    D = np.loadtxt('data/basketsenum2017.txt',dtype=int,delimiter=',')
    D = sp.csr_matrix(D)
    pcy(D,152,2)