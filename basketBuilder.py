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
                if magnitudeChange < 0.5:
                    magnitudeData = dayData.get('0-.499', [])
                    magnitudeData.append(stockName)
                    dayData.update({'0-.499' : magnitudeData})
                elif magnitudeChange < 1.0:
                    magnitudeData = dayData.get('.5-.999', [])
                    magnitudeData.append(stockName)
                    dayData.update({'.5-.999' : magnitudeData}) 
                elif magnitudeChange < 1.5:
                    magnitudeData = dayData.get('.999-1.49', [])
                    magnitudeData.append(stockName)
                    dayData.update({'.999-1.49' : magnitudeData})
                elif magnitudeChange < 2.0:
                    magnitudeData = dayData.get('1.49-1.99', [])
                    magnitudeData.append(stockName)
                    dayData.update({'1.49-1.99' : magnitudeData})
                elif magnitudeChange < 2.5:
                    magnitudeData = dayData.get('1.99-2.49', [])
                    magnitudeData.append(stockName)
                    dayData.update({'1.99-2.49' : magnitudeData})
                elif magnitudeChange < 3.0:
                    magnitudeData = dayData.get('2.49-2.99', [])
                    magnitudeData.append(stockName)
                    dayData.update({'2.49-2.99' : magnitudeData})
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
            

            
            

        
            