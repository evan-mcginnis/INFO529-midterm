#
# I N F O  5 2 9  M I D T E R M
#
# M A K E F I L E
#

# The names of the files. These can be changed
TESTDATA=test
TRAINDATA=train
MODEL=model_midterm.ckpt
usage:
	echo "No target specified. Use one of test-data or train-data"


data: Dataset_Competition
	wget https://data.cyverse.org/dav-anon/iplant/home/evanmc/Dataset_Competition_Zip_File.zip
	unzip Dataset_Competition_Zip_File.zip

# How to convert a .csv to a .npz
%.npz: %.csv
	python csv2npz.py -i $< -o $@

$(TRAINDATA).csv:
	python makecsv.py -w "Dataset_Competition/Test Inputs/inputs_weather_test.npy" -o "Dataset_Competition/Test Inputs/inputs_others_test.npy" -g "Dataset_Competition/clusterID_genotype.npy" -o "Dataset_Competition/Test Inputs/inputs_others_test.npy" -c $(TRAINDATA).csv

$(TRAINDATA).npz: $(TRAINDATA).csv

train-data: $(TRAINDATA).npz

$(TESTDATA).csv:
	python makecsv.py -w "Dataset_Competition/Test Inputs/inputs_weather_test.npy" -o "Dataset_Competition/Test Inputs/inputs_others_test.npy" -g "Dataset_Competition/clusterID_genotype.npy" -c $(TESTDATA).csv

$(TESTDATA).npz: $(TESTDATA).csv

# Use this rule to make the test data with 'make test-data'
test-data: $(TESTDATA).npz

# Predict from test data
predictions: $(TESTDATA).npz
	python predict-from-model.py -m $(MODEL) -d $(TESTDATA).npz
