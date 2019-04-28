from os import walk
import numpy as np

if __name__ == "__main__":
    stockBaskets = {} 
    for root, dirs, filenames in walk('StocksCleaned/'):
        for stock in filenames:
            stockName = str(stock).split('D')[0]
            stockData = np.loadtxt(fname = 'StocksCleaned/' + str(stock), delimiter = ',', dtype = str)
            for dataPoint in stockData:
                dayData = stockBaskets.get(str(dataPoint[0]), {})
                magnitudeChange = abs(dataPoint[1].astype(float))
                if magnitudeChange < 0.25:
                    magnitudeData = dayData.get('0-.249', [])
                    magnitudeData.append(stockName)
                    dayData.update({'0-.249' : magnitudeData})
                elif magnitudeChange < .5:
                    magnitudeData = dayData.get('.25-.499', [])
                    magnitudeData.append(stockName)
                    dayData.update({'.25-.499' : magnitudeData}) 
                elif magnitudeChange < .75:
                    magnitudeData = dayData.get('.5-.749', [])
                    magnitudeData.append(stockName)
                    dayData.update({'.5-.749' : magnitudeData})
                elif magnitudeChange < 1.0:
                    magnitudeData = dayData.get('.75-.99', [])
                    magnitudeData.append(stockName)
                    dayData.update({'.75-.99' : magnitudeData})
                elif magnitudeChange < 1.25:
                    magnitudeData = dayData.get('1.0-1.249', [])
                    magnitudeData.append(stockName)
                    dayData.update({'1.0-1.249' : magnitudeData})
                elif magnitudeChange < 1.5:
                    magnitudeData = dayData.get('1.25-1.499', [])
                    magnitudeData.append(stockName)
                    dayData.update({'1.25-1.499' : magnitudeData}) 
                elif magnitudeChange < 1.75:
                    magnitudeData = dayData.get('1.5-1.749', [])
                    magnitudeData.append(stockName)
                    dayData.update({'1.5-1.749' : magnitudeData})
                elif magnitudeChange < 2.0:
                    magnitudeData = dayData.get('1.75-1.99', [])
                    magnitudeData.append(stockName)
                    dayData.update({'1.75-1.99' : magnitudeData})
                elif magnitudeChange < 2.25:
                    magnitudeData = dayData.get('2.00-2.249', [])
                    magnitudeData.append(stockName)
                    dayData.update({'2.00-2.249' : magnitudeData})
                elif magnitudeChange < 2.5:
                    magnitudeData = dayData.get('2.25-2.499', [])
                    magnitudeData.append(stockName)
                    dayData.update({'2.25-2.499' : magnitudeData}) 
                elif magnitudeChange < 2.75:
                    magnitudeData = dayData.get('2.5-2.749', [])
                    magnitudeData.append(stockName)
                    dayData.update({'2.5-2.749' : magnitudeData})
                elif magnitudeChange < 3.0:
                    magnitudeData = dayData.get('2.75-2.99', [])
                    magnitudeData.append(stockName)
                    dayData.update({'2.75-2.99' : magnitudeData})
                else:
                    magnitudeData = dayData.get('3.00+', [])
                    magnitudeData.append(stockName)
                    dayData.update({'3.00+' : magnitudeData})                            
                stockBaskets.update({str(dataPoint[0]) : dayData})
    # print(stockBaskets)
    f = open("baskets.txt", "a+")
    for key in stockBaskets.keys():
        for secondKey in sorted(stockBaskets.get(key)):
            # f.write(str(key) + ", " +  str(secondKey) + ": " + str(stockBaskets.get(key).get(secondKey)))
            f.write(str(stockBaskets.get(key).get(secondKey)) + "\n")
            

            
            

        
            