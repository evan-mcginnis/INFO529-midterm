import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.cluster import KMeans
import sys

parser = argparse.ArgumentParser("INFO 529 Midterm head")

parser.add_argument('-o', '--others', action="store", required=True, help="Others")
parser.add_argument('-w', '--weather', action="store", required=True, help="Weather")
parser.add_argument('-y', '--yield', action="store", required=True, help="Yield", dest="yld")

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
y = np.load(arguments.yld)

if (o is None or w is None or y is None):
    print("Failed to load data")
    sys.exit(-1)

MATURITY = "Maturity"
GENOTYPE = "Genotype"
STATE = "State"
YEAR = "Year"
LOCATION = "Location"

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

weather = pd.DataFrame({WEATHER_ADNI: [w[:,0]],
                        WEATHER_AP: [w[:,1]],
                        WEATHER_ARH: [w[:,2]],
                        WEATHER_MNDI: [w[:,3]],
                        WEATHER_MAX_TEMP: [w[:,4]],
                        WEATHER_MIN_TEMP: [w[:,5]],
                        WEATHER_AVG_TEMP: [w[:,6]]})

yld = pd.DataFrame({})


states = others.groupby(by=STATE)
#print("State counts\n{}".format(states.count()))
print("States")
print(states.head())

maturity = others.groupby(by=MATURITY)
print("Maturity")
print(maturity.head())

# Well this is irritating. State names have quotes in them
# And this is probably an error when I built the frame.  The year is a string, not an integer
#maturity = others.query('Maturity == 6.0 & Year == 2015.0')
maturity = others.query('Year == 2015.0')

exit(rc)
kmeans = KMeans(n_clusters=20, random_state=0).fit(maturity[GENOTYPE])
print(kmeans.labels_)

yields = []
for entry in maturity.index:
    yields.append(y[entry])

xaxis = np.arange(start=0,stop=len(maturity[GENOTYPE]))
plt.figure()
plt.scatter(xaxis, maturity[GENOTYPE])
plt.show()

xaxis = np.arange(start=0,stop=len(yields))

plt.figure()
plt.plot(yields)
plt.show()

print("There are {} unique genotypes in this maturity".format(maturity.nunique()))
# for entry in maturity.index:
#     print(y[entry])


exit(rc)