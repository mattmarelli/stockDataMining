import numpy as np
from os import walk



if __name__ == "__main__":
    startYear = 1999
    endYear = 2000
    for root, dirs, filenames in walk('Stocks/Stocks/'):
        for stock in filenames:
            stockData = np.loadtxt(fname = 'Stocks/Stocks/' + str(stock), delimiter = ',', dtype = str)
            stockData = stockData[1:]
            date = stockData[:,0]
            # firstDate = date[0]
            # firstYear = firstDate[:4]
            # if int(firstYear) == startYear + 1:
            count = 0
            validYear = False
            oneDay = ""
            tempList = []
            for lineOfData in np.nditer(stockData):
                if count % 7 == 0:
                    oneDay = str(lineOfData)
                    year = oneDay[:4]
                    if int(year) < endYear + 1 and int(year) > startYear:
                        validYear = True
                    else:
                        validYear = False
                else:
                    if validYear:
                        if count % 7 == 1:
                            openingPrice = str(lineOfData)
                        if count % 7 == 4:
                            closingPrice = str(lineOfData)
                            dayChange = float(closingPrice) - float(openingPrice)
                            percentChange = (dayChange / float(openingPrice)) * 100
                            tempList.append([oneDay, percentChange])
                        # if count % 7 == 6:
                        #     dayChange = float(closingPrice) - float(openingPrice)
                        #     percentChange = (dayChange / float(openingPrice)) * 100
                        #     tempList.append([oneDay, percentChange])
                count += 1
            if tempList != []:
                outputList = np.array(tempList)
                output = np.zeros(shape= (len(tempList), 2)).astype(str)
                output[:,0] = outputList[:,0].astype(str)
                output[:,1] = outputList[:,1].astype(str)
                # temp = stock.split('.')
                fileName = "StocksCleaned/" + stock.split('.')[0] + 'DailyChange.txt'
                np.savetxt(fname = fileName, X = output, fmt='%s, %s', delimiter= ',', encoding = 'utf8')