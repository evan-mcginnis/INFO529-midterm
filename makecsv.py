import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
#from sklearn.cluster import KMeans
import sys

parser = argparse.ArgumentParser("INFO 529 Create CSV")

parser.add_argument('-o', '--others', action="store", required=True, help="Others")
parser.add_argument('-w', '--weather', action="store", required=True, help="Weather")
parser.add_argument('-y', '--yield', action="store", required=False, help="Yield -- required only for training data", dest="yld")
parser.add_argument('-g', '--genotype', action="store", required=True, help="Genotype Mapping")
parser.add_argument('-c', '--csv', action="store", required=True, help="CSV file")

arguments = parser.parse_args()

rc = 0

def load(name: str) -> np.ndarray:
    d = None
    try:
        d = np.load(name)
    except FileNotFoundError as e:
        print(e)
    return d

o = np.load(arguments.others)
w = np.load(arguments.weather)
g = np.load(arguments.genotype)

# For the test data production, the yield would not be given
# So this is optional.  I suppose I could introduce a 'type' so we could make this
# conditionally optional

y = None

# The type of output file we are making.
# 3 and 2 may seem arbitrary, but are used below as the number of attributes to expect

TYPE_TRAINING = 3
TYPE_TEST = 2
TYPE = TYPE_TEST

if arguments.yld is not None:
    TYPE = TYPE_TRAINING
    y = np.load(arguments.yld)
    if y is None:
        print("Failed to load data")
        sys.exit(-1)

if (o is None or w is None or g is None):
    print("Failed to load data")
    sys.exit(-1)

MATURITY = "Maturity"
GENOTYPE = "Genotype"
STATE = "State"
YEAR = "Year"
LOCATION = "Location"
CLUSTER = "Cluster"

YIELD = "Yield"

WEATHER_ADNI = "ADNI"
WEATHER_AP = "AP"
WEATHER_ARH = "ARH"
WEATHER_MNDI = "MNDI"
WEATHER_MAX_TEMP = "MaxSur"
WEATHER_MIN_TEMP = "MinSur"
WEATHER_AVG_TEMP = "AvgSur"

# Construct data frames
others = pd.DataFrame({MATURITY: o[:,0].astype(float),
                       GENOTYPE : o[:,1].astype(float),
                       STATE : o[:,2],
                       YEAR : o[:,3].astype(float),
                       LOCATION: o[:,4].astype(float)})

weather = pd.DataFrame({WEATHER_ADNI: w[:,0].flatten(),
                        WEATHER_AP: w[:,1].flatten(),
                        WEATHER_ARH: w[:,2].flatten(),
                        WEATHER_MNDI: w[:,3].flatten(),
                        WEATHER_MAX_TEMP: w[:,4].flatten(),
                        WEATHER_MIN_TEMP: w[:,5].flatten(),
                        WEATHER_AVG_TEMP: w[:,6].flatten()})

yld = pd.DataFrame({})

print("Number of unique genotypes: {}".format(others.Genotype.nunique()))
print("Size of genotype mapping data: {}".format(len(g)))

def genotypeCluster(genotype: int) -> int:
    return(g[int(genotype) - 1])

def enrich_data(others: pd.DataFrame, genotypeMapping: np.ndarray) -> pd.DataFrame:
    others.insert(2, CLUSTER, 0)
    others[CLUSTER] = others.apply(lambda x: genotypeCluster(x[GENOTYPE]), axis=1)
    #self.worksheet_for_eto[self.DOG] = self.worksheet_for_eto.apply(lambda row: self.__day_of_growth(row['DOY']), axis=1)
    return others

others = enrich_data(others, g)

# This shows how to pull out the weather data for a single location
# weatherForOneLocation = w[0,:,:]
# xaxis = np.arange(start=0, stop=len(weatherForOneLocation))
# plt.figure()
# # And this is a plot of a specific weather reading -- 5 is average temp.
# plt.plot(xaxis, weatherForOneLocation[:,4], label="min")
# plt.plot(xaxis, weatherForOneLocation[:,5], label="max")
# plt.plot(xaxis, weatherForOneLocation[:,6], label="avg")
# plt.legend()
# plt.xlabel("days in season")
# plt.ylabel("temperature")
# plt.show()


mergedData = pd.DataFrame()
mergedData[LOCATION] = others[LOCATION]
mergedData[YEAR] = others[YEAR]

# Add in yield if it is there
if y is not None:
    mergedData[YIELD] = y

# Construct the merged array.  Sure there is a more elegant way to do this
# Weather is 93K x 214 days x 7 weather attributes
for i in range(0, 1): # We just want to do this once, so we don't need w.shape[0]):
    for day in range(0, w.shape[1]):
        for attribute in range(0, w.shape[2]):
            name = "w_" + str(day) + "_" + str(attribute)
            mergedData.insert(mergedData.shape[1],name,0)

for i in range(0, w.shape[0]):
    print("Row {}".format(i))
    for day in range(0, w.shape[1]):
        for attribute in range(0, w.shape[2]):
            name = "w_" + str(day) + "_" + str(attribute)
            #mergedData[i,day*w.shape[2] + attribute + 3] = w[i,day,attribute]
            # If we are producing training data, there are 3 columns before the reading
            # If we are producing test data, there are 2.  Either way, we get this from TYPE
            mergedData.iat[i,day*w.shape[2] + attribute + TYPE] = w[i,day,attribute]

# Avoid getting an index column in the CSV
mergedData.to_csv(arguments.csv, index=False)

exit(rc)

# Leftover data exploration items
# Leave these in case we need them,,,
#
# kmeans = KMeans(n_clusters=20, random_state=0).fit(maturity[GENOTYPE])
# print(kmeans.labels_)
#
# yields = []
# for entry in maturity.index:
#     yields.append(y[entry])
#
# xaxis = np.arange(start=0,stop=len(maturity[GENOTYPE]))
# plt.figure()
# plt.scatter(xaxis, maturity[GENOTYPE])
# plt.show()
#
# xaxis = np.arange(start=0,stop=len(yields))
#
# plt.figure()
# plt.plot(yields)
# plt.show()
#
# print("There are {} unique genotypes in this maturity".format(maturity.nunique()))
# # for entry in maturity.index:
# #     print(y[entry])


exit(rc)