"""
Run as a process
"""

# Import friends
import pandas as pd
import numpy as np
from re import sub

# Import prediction related stuffs from scikit-learn
from sklearn import svm, grid_search
from sklearn import preprocessing as pre

import zmq
import json

# Date related stuffs
from datetime import datetime, date

# Logging stuffs
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoadPredictor(object):

	def __init__(self, granular=15, trainingPeriods=28, trainingWindow=7):

		# Set granularity and related stuffs
		self.__hour = 60 # Mins
		self.__granular = granular # Mins

		# Periods in a day
		self.periods = 24 * int(self.__hour/self.__granular)
		
		# Training periods
		self.__priorPeriods = self.periods * trainingPeriods
		
		# Training window
		self.__training = self.periods * trainingWindow

		# Define sampling frequency
		self.__frequency = '%dT'%self.__granular

		# Initialize the input data list
		self.__input_data = dict()

		# Default hyper-parameters for SVM
		# will be changed (possibly) after gridsearch
		self.__C = 100000.0
		self.__gamma = 1e-5

		# Perform the costly grid search (only once though)
		self.__performGridSearch = False


	def input(self, _type='historic_data', _data=None):
		# Assuming the _data will be as Pandas dataframe
		self.__input_data.update({_type: _data})

		# Prepare the input data
		self.__prepareData()


	def __prepareData(self):

		# Grab different types of input data
		self.__input_types = list(self.__input_data.keys())

		# Raw data
		self.__raw_data = []
		for k, v in self.__input_data.items():
			self.__raw_data.append(np.array(v))

		"""Generate the temporal data that might influence the signal (e.g. load)"""
		dt_index = self.__input_data[self.__input_types[0]].index

		# Day of week		
		self.__day_of_week = np.array(dt_index.dayofweek)

		# Period of the day (i.e. 0...23 for 1 hour granularity, 0...47 for 30 mins granularity)
		self.__pd_of_day = np.array(dt_index.hour * 
							int(self.__hour/self.__granular) + 
							dt_index.minute/self.__granular)
		# Working month
		self.__working_mnt = np.array(dt_index.month)

		# Just place the holiday (for now only weekends) info
		holiday_info = np.zeros(len(dt_index))
		holiday_info[(dt_index.dayofweek==5)|(dt_index.dayofweek==6)] = 1
		self.__holiday = np.array(holiday_info)


	def __preparePredictionData(self, timeIndex):
		""" Prepare the x data for prediction i.e.temporal information
		such as, days, periods, months, and holidays.
		"""
		pLoad = np.zeros(len(timeIndex))
		pdf = pd.DataFrame(pLoad, index=timeIndex,)

		# Day of week		
		self.__p_day_of_week = np.array(pdf.index.dayofweek)

		# Period of the day (i.e. 0...23 for 1 hour granularity, 
		# 0...47 for 30 mins granularity or 0...95 for 15 mins granularity)
		self.__p_pd_of_day = np.array(pdf.index.hour * 
							int(self.__hour/self.__granular) + 
							pdf.index.minute/self.__granular)
		# Working month
		self.__p_working_mnt = np.array(pdf.index.month)

		# Just place the holiday (for now only weekends) info 
		pdf['holiday'] = np.zeros(len(pdf))
		pdf['holiday'][(pdf.index.dayofweek==5)|(pdf.index.dayofweek==6)] = 1
		self.__p_holiday = np.array(pdf['holiday'])


	
	def __bestHParameter(self, x_search, y_search):
		""" Performing grid search to findout optimal
			hyper-parameter for SVR. [C and Gamma]

		Returns:
			Dictionary of hyper-parameters

		"""
		print("Performing grid search")

		parameters = {'C':[.001, .01, .1, 1, 10, 100, 1e+4, 1e+5, 1e+6], 
					  'gamma':[1e-7, 1e-6, .000001, .00001, .0001, .001, .01, 
							   .1, 1, 10, 100, 1000, 1e+4, 1e+5, 1e+6]}

		gs_SVR_model = svm.SVR(kernel='rbf')
		SVR_model_optParams = grid_search.GridSearchCV(gs_SVR_model, parameters, n_jobs=4)

		SVR_model_optParams.fit(x_search, y_search)

		return SVR_model_optParams.best_params_


	def predict(self, t="2015-01-15 00:30:00", window=None,):

		""" predict the demand/signal (wrapper to predictBySVR method)
		"""
		
		# If no window is given, use the defualt periods_per_day value
		if window is None:
			window = self.periods

		# Finding equivalent datetime
		startDateTime = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')

		# Find the data index
		t = len(self.__input_data[self.__input_types[0]][:startDateTime]) - 1

		# Create the datetime index
		dt = pd.date_range(startDateTime, periods=window, freq=self.__frequency)

		# Create prediction temporal information with the datetime index
		self.__preparePredictionData(timeIndex = dt)

		# Predict the load
		PD = self.__predictBySVR(t, window)
		# PD = self.__predictByAverage(dt)

		# Create the dataframe
		df = pd.DataFrame(PD, index = dt, columns=['load_prediction'])

		return df


	def __predictByAverage(self, dt):
		""" Preform prediction just by taking the periodic average
			of the historical data.
		"""	
		
		# Have the historical data ready
		past_data = self.__input_data['historic_data']
		
		# list of predictions
		predict = []

		# Iterate over time
		for tm in dt:
			# Timeindex as a function of granularity
			ti = tm.hour * int(self.__hour/self.__granular) + tm.minute/self.__granular

			# Calculate the mean of the historical load for same period
			load = past_data[past_data.index.hour * int(self.__hour/self.__granular) + past_data.index.minute/self.__granular==ti].mean()
			
			predict.append(load)
			logger.info("Predicted {} for period {}.".format(load, str(tm)))

		return predict


	def __predictBySVR(self, t, window):
		""" Perform prediction.
		Currently, the training the models and prediction are put together.
		Later they will be seperated.

		"""
		# Start of the training
		start = t - self.__training

		# list of prediction
		predict = []
		
		# Record the starting t        
		start_t = t

		# Historical (demand) data upto the current time , t (make sure not to 
		# build the predictive model utilizing the actual data!!)
		pastSignal = self.__raw_data[0]

		# For each prediction window, create an SVM model and predict the next period		
		for pd in range(window):

			# Generate the training data, x is the input [as past demands]
			x_train = [[pastSignal[i-k-pd] for k in range(self.__priorPeriods+1)[1:]] + \
						[self.__working_mnt[i], self.__day_of_week[i], self.__pd_of_day[i], self.__holiday[i]] \
						for i in range(start, t-pd)]

			# training data, y is the output [as past demands]
			y_train = [pastSignal[i] for i in range(start, t-pd)]

			# Transform data to be normally distributed with zero mean and unit variance
			scaler = pre.StandardScaler().fit(x_train)
			x_train_scaled = scaler.transform(x_train)   

			# Finding the best parameters via grid search [Perform at the initial stage]
			if self.__performGridSearch:
				bp = self.__bestHParameter(x_train_scaled, y_train)

				# Store the best parameters
				self.__C = bp['C']
				self.__gamma = bp['gamma']

				logger.info("Best hyperparameters: {}, {}".format(self.__C, self.__gamma))

				# stop performing gridsearch next time
				self.__performGridSearch = False

			# Fit the model with SVR.
			# Later 
			svr_model = svm.SVR(kernel='rbf', C=self.__C, gamma=self.__gamma).fit(x_train_scaled, y_train)

			# Prepare the prediction input data
			x_test = [[pastSignal[i-k-pd] for k in range(self.__priorPeriods+1)[1:]] + \
						[self.__p_working_mnt[i - start_t], self.__p_day_of_week[i - start_t], 
						 self.__p_pd_of_day[i - start_t], self.__p_holiday[i - start_t]] \
						for i in range(t, t+1)]        

			# Scale the testing input using the same scaler as training input
			x_test_scaled  = scaler.transform(x_test)

			# Predict the demand
			y_predict = svr_model.predict(x_test_scaled)
			
			# localize the value
			pValue = y_predict[0]

			# Place it into the predictor list
			predict.append(pValue)            

			logger.info("Predicted {} for period {}.".format(pValue, t))

			# Go next
			t += 1
		
		# Rerturn the list
		return predict

def main():
		
		# Start the load prediction as a process
		# that is running in a host using zmq.
		# REQ-REP pattern for now.
		load_predictor_host = "tcp://127.0.0.1:9090"

		# Open a context for zmq and bind a socket with it
		# The predictions server supposed to run on it

		context = zmq.Context()
		socket = context.socket(zmq.REP)
		socket.bind(load_predictor_host)

		logger.info("Predictor is running on {}".format(load_predictor_host))

		while True:
			# Get the input from a client
			network_serialized = socket.recv_json()

			# Retrieve the dictionary
			network_data = json.loads(network_serialized)

			logger.info(network_data.keys())			

			# localize
			starting_datetime = network_data['starting_datetime']
			prediction_window = network_data['prediction_window']

			# unserialize (historic data)
			input_data = pd.read_json(network_data['input_data'])

			# Instantiate the predictor class
			lp = LoadPredictor()

			# Relay the input (for now, only historicla data)
			lp.input(_type='historic_data', _data=input_data['load'])

			# Perform the prediction (training the models will be done inside the .predict method)
			prediction_df = lp.predict(t=str(starting_datetime), window=prediction_window)

			# Send the prediction back to 
			socket.send_json(prediction_df.to_json())

if __name__ == '__main__':
	main()