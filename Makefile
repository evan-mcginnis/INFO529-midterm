#
# I N F O  5 2 9  M I D T E R M
#
# M A K E F I L E
#

# The names of the files. These can be changed
TESTDATA=test
TRAINDATA=train

usage:
	echo "No target specified. Use one of test-data or train-data"

# How to convert a .csv to a .npz
%.npz: %.csv
	python csv2npz.py -i $< -o $@

train-data:
	python makecsv.py -w "Dataset_Competition/Test Inputs/inputs_weather_test.npy" -o "Dataset_Competition/Test Inputs/inputs_others_test.npy" -g "Dataset_Competition/clusterID_genotype.npy" -o "Dataset_Competition/Test Inputs/inputs_others_test.npy" -c test.csv
	$(TRAIN).npz


$(TESTDATA).csv:
	python makecsv.py -w "Dataset_Competition/Test Inputs/inputs_weather_test.npy" -o "Dataset_Competition/Test Inputs/inputs_others_test.npy" -g "Dataset_Competition/clusterID_genotype.npy" -c $(TESTDATA).csv

$(TESTDATA).npz: $(TESTDATA).csv

# Use this rule to make the test data with 'make test-data'
test-data: $(TESTDATA).npz