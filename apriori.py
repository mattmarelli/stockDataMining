from itertools import chain
from itertools import combinations
import numpy as np


def read_dataset(filename):
    f = open(filename, 'r')
    dataset = []
    for line in f.readlines():
        dataset.append(line.replace(' \n', '').split(' '))
    f.close()
    return dataset


# takes a tuple T and returns a hashed index
def pcy_hash(T,n):
    # TODO perhaps write a better hash 
    index = sum([hash(t) for t in T])
    index = index % n
    return index


#returns all possible combinations 
# within array A of size n
def combinations(A,n):
    combos = chain.from_iterable(combinations(d,n))
    combos = np.fromiter(combos,dtype=str)
    combos = combos.reshape((len(combos) // n),n)
    return combos


# Associative rule mining for dataset D with
# and s is support for k tuples
def pcy(D,s,k):
    n = len(np.unique(D))
    buckets = np.zeros(
        shape=(n//2), dtype=int)  
    bitmap = None

    # helper function increments buckets
    # given an array of hash values
    @np.vectorize
    def increment_buckets(index):
        buckets[index] += 1

    # loop through finding frequent 
    # k-tuples of support s
    for i in range(1,k+1):
        for basket in D:
            # generate canidate tuples
            canidates = combinations(basket,i)
            for candidate in canidates:
                combos = combinations(candidate,(i-1))
                # hash each combination within candidate to check if they
                indices = np.where(combos,pcy_hash(combos,(n // i)),-1)


            # hash each canidate tuple in canidates and increment bucket
            
            increment_buckets(hashed)
            bitmap = np.where(buckets > s,1,0)


# TODO additional preprocessing dataset
# encode stocks to np.uint16 index
if __name__ == '__main__':
    # set support to 860 such that stocks are similar to
    # are 70 percent of the days in the last five years
    D = read_dataset('data/browsingdata_50baskets.txt')
    pcy(D,4,2)