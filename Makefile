#
# I N F O  5 2 9  M I D T E R M
#
# M A K E F I L E
#

# The names of the files. These can be changed
TESTDATA=test
TRAINDATA=train
MODEL=model_midterm
MPATH=model
DATADIR=./data

# The number of iterations in training
ITERATIONS=350000

usage:
	echo "No target specified. Use one of test-data, train-data, train, or predict"


$(DATADIR):
	@mkdir -p $@

data-for-competition: | $(DATADIR)
	wget https://data.cyverse.org/dav-anon/iplant/home/evanmc/Dataset_Competition_Zip_File.zip
	# Need unkzip for cygwin to debug this on windows.
	#unzip Dataset_Competition_Zip_File.zip
	wget https://data.cyverse.org/dav-anon/iplant/home/evanmc/train.npz -P $(DATADIR)
	wget https://data.cyverse.org/dav-anon/iplant/home/evanmc/test.npz -P $(DATADIR)


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
	python predict-yield.py -p $(MPATH) -m $(MODEL) -i $(ITERATIONS) -d $(TRAINDATA).npz

# Predict from test data
predictions: $(TESTDATA).npz
	python predict-from-model.py -m $(MODEL) -d $(TESTDATA).npz

docker: 
	docker build -t midterm .

# Super-bad form to put the password here, but we want this to go to my account
docker-push:
	docker login -u emcginnis -p AZj7X7jyzf9W
	docker push emcginnis/info529:latest

all: data-for-competition train-data test-data train predictions
	@echo Complete

clean:
	rm -rf Dataset_Competition
	rm $(DATADIR)/$(TRAINDATA).npz
	rm $(DATADIR)/$(TESTDATA).npz
	rm $(MODEL).*
