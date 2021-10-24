#
# I N F O  5 2 9  M I D T E R M
#
# M A K E F I L E
#

# The names of the files. These can be changed
TESTDATA=test
TRAINDATA=train
MODEL=model_midterm.ckpt

# The number of iterations in training
ITERATIONS=350000

usage:
	echo "No target specified. Use one of test-data, train-data, train, or predict"


data:
	wget https://data.cyverse.org/dav-anon/iplant/home/evanmc/Dataset_Competition_Zip_File.zip
	unzip Dataset_Competition_Zip_File.zip

# How to convert a .csv to a .npz
%.npz: %.csv
	python csv2npz.py -i $< -o $@

$(TRAINDATA).csv:
	python makecsv.py -w "Dataset_Competition/Training/inputs_weather_train.npy" -o "Dataset_Competition/Training/inputs_others_train.npy" -g "Dataset_Competition/clusterID_genotype.npy" -y "Dataset_Competition/Training/yield_train.npy" -c $(TRAINDATA).csv

$(TRAINDATA).npz: $(TRAINDATA).csv

train-data: $(TRAINDATA).npz

$(TESTDATA).csv:
	python makecsv.py -w "Dataset_Competition/Test Inputs/inputs_weather_test.npy" -o "Dataset_Competition/Test Inputs/inputs_others_test.npy" -g "Dataset_Competition/clusterID_genotype.npy" -c $(TESTDATA).csv

$(TESTDATA).npz: $(TESTDATA).csv

# Use this rule to make the test data with 'make test-data'
test-data: $(TESTDATA).npz

# This is how you train and then export the model
train: $(TRAINDATA).npz
	python predict-yield.py -m $(MODEL) -i $(ITERATIONS) -d $(TRAINDATA).npz

# Predict from test data
predictions: $(TESTDATA).npz
	python predict-from-model.py -m $(MODEL) -d $(TESTDATA).npz

all: data train-data test-data train predictions
	@echo Complete

clean:
	rm -rf Dataset_Competition
	rm $(TRAINDATA).npz
	rm $(TESTDATA).npz
	rm $(MODEL).*