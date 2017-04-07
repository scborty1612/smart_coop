"""
Scheduler class:
Later it could be implemented as a process.
For now, just bind it with the agent class.
"""

# Import friends
import pandas as pd
import numpy as np
from re import sub

import sys
sys.path.append("../")
from configure import Configure as CF

# import zmq
# import json

from pulp import *

# Date related stuffs
from datetime import datetime, date

# Logging stuffs
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Scheduler(object):

	def __init__(self, agent_id=0, granular=15, periods=None, batteryInfo=None, init_soc=0.8,
					predicted_demand=None,
					predicted_gen=None):

		# Record the agent ID (may not be available when run as a proecess)
		self.agent_id = agent_id

		# Set granularity and related stuffs
		self.__hour = 60 # Mins
		self.__granular = granular # Mins

		# Delta T
		self.__delT = self.__granular/self.__hour

		# Periods in a day
		if periods is None:
			self.periods = 24 * int(self.__hour/self.__granular)
		else:
			self.periods =periods 
		
		if batteryInfo is None:
			self.__batteryInfo = CF.BATTERY_CHAR
		else:
			self.__batteryInfo = batteryInfo

		# Battery information
		self.init_soc = init_soc

		# Load the predicted demand and generation
		self.predicted_demand = np.zeros(self.periods + 1)
		self.predicted_demand[1:] = predicted_demand

		self.predicted_gen = np.zeros(self.periods + 1)
		self.predicted_gen[1:] = predicted_gen

		# The size of the predictions equals number of periods
		# They should be the numpy vectors
		# Save the 0-th index for initial state
		self.revised_demand = self.predicted_demand - self.predicted_gen

		# Initialize the resultant battery power
		self.battery_power = np.zeros(periods+1)

		# Initalize the SOC
		self.__init_soc = init_soc


	def optimize(self, ):

		#d = 40
		# Constant degradation factor for the battery
		degrad_battery = 0.02
		
		# Epson to replace zero
		epson = 1e-7
		
		# Total daily load (that will not be changed)
		# The optimizer will not try to minimize the total demand,
		# instead schedule the battery that will minimize the load factor (peak to average ratio)
		total_demand = np.sum(self.predicted_demand)

		# Derive the linear programming based formulation
		batteryProb = LpProblem("BatteryScheduling", LpMinimize)
		
		# time index
		tIn = [i for i in range(self.periods+1)]
		
		# Define the decision variables
		
		# Dispatched battery power [Real variable; continuous]
		Pb = pulp.LpVariable.dicts("batteryDispatch", (tIn))
		
		# Battery energy status []
		Xb = pulp.LpVariable.dicts("batteryEnergyStatus", (tIn), lowBound = 0)
		
		# Battery status variable (0 charge, 1 discharge) [Binary]
		Sb = pulp.LpVariable.dicts("batteryChargeDischargeStatus", (tIn), cat=pulp.LpBinary)
		 
		# Auxiliary variables to handle logical (in)equivalences
		Ax = pulp.LpVariable.dicts("auxEnergyStatus", (tIn), cat=pulp.LpContinuous)
		
		# Handling the max(k)
		J = pulp.LpVariable("J", cat=pulp.LpContinuous)

		
		# Objective function [Minimizing Periods*J]        
		batteryProb +=  self.periods * J
		
		
		"""Constraints and model definition"""
		# Model initialization
		batteryProb += (Xb[0] == self.__batteryInfo['capacity'] * self.__init_soc)
		
		# Initialize the battery C/D status (always starts by discharging the battery)
		batteryProb += (Sb[0] == 0)

		# For each, have the battery constraints ready
		for t in range(self.periods+1)[1:]:
			
			# Power balance i.e. total load is equivalent to total generation (Battery+PV)
			batteryProb += (Pb[t]  == self.predicted_demand[t] - self.predicted_gen[t]), "power_balance_at_period_%d"%t
			
			# Battery operation (changes in SOC by charging or discharging the battery with corresponding efficiencies)
			batteryProb += (Xb[t] == Xb[t-1] + \
				((self.__batteryInfo['c_eff']  - 1/self.__batteryInfo['d_eff']) * Ax[t] + \
					1/self.__batteryInfo['d_eff'] * Pb[t]) * self.__delT - degrad_battery), "battery_soc_change_at_period_%d"%t
				
			# The battery energy level should be within the SOC limits
			batteryProb += (Xb[t] <= self.__batteryInfo['capacity'] * self.__batteryInfo['soc_high']),"battery_energy_up_at_period_%d"%t
			batteryProb += (Xb[t] >= self.__batteryInfo['capacity'] * self.__batteryInfo['soc_low'] ),"battery_energy_down_at_period_%d"%t
			
			# Followings are additional MIP constraints to handle inequalities in SOC dynamics

			# Discharge 
			batteryProb += (   self.__batteryInfo['d_rating']             * Sb[t]  + ( 0)* Ax[t]  - ( 1) * Pb[t] - self.__batteryInfo['d_rating'] <= 0)
			batteryProb += (- (self.__batteryInfo['d_rating'] + epson)    * Sb[t]  + ( 0)* Ax[t]  - (-1) * Pb[t] - (-epson)                   <= 0)
			batteryProb += (   self.__batteryInfo['d_rating']             * Sb[t]  + ( 1)* Ax[t]  - ( 1) * Pb[t] - self.__batteryInfo['d_rating'] <= 0)
			batteryProb += (   self.__batteryInfo['d_rating']             * Sb[t]  + (-1)* Ax[t]  - (-1) * Pb[t] - self.__batteryInfo['d_rating'] <= 0)
			
			# Charge (typically in this case, the battery is charged with the maximum charing power available)            

			# batteryProb += ( Ax[t] == self.__batteryInfo['c_rating'] * Sb[t])

			batteryProb += (-  self.__batteryInfo['c_rating']  * Sb[t]  + ( 1)* Ax[t]  - ( 0) * Pb[t] - 0 <= 0)
			batteryProb += (-  self.__batteryInfo['c_rating']  * Sb[t]  + (-1)* Ax[t]  - ( 0) * Pb[t] - 0 <= 0)
		
		# constraining J
		for t in range(self.periods+1):
			batteryProb += (J >= self.revised_demand[t] + Pb[t]), "max_J_at_%d"%t
		
		# total load should not change
		batteryProb += (sum(self.revised_demand[t]+Pb[t] for t in range(self.periods+1)[1:]) == total_demand), "total_load_to_be_same"

		# Solve the battery problem using COIN-R 
		batteryProb.writeLP("problem_model_{}.lp".format(self.agent_id))
		
		batteryProb.solve(COIN(msg = 0, threads=4))
		
		# Check the status (anything except 'optimal' is not expected)
		logging.info(pulp.LpStatus[batteryProb.status])
		
		# Record the total cost (approximated)
		self.__ObjectiveValue = value(batteryProb.objective)
		
		# Record the decision variables/status/parameter
		self.__BatteryEnergyStatus = np.zeros(self.periods + 1)        
		self.__BatterySOC          = np.zeros(self.periods + 1)
		self.__BatteryCDStatus     = np.zeros(self.periods + 1)
		
		# Record the variables with values
		for v in batteryProb.variables():             
			if  "batteryEnergyStatus" in v.name:
				period = int(v.name.split("_")[1])
				self.__BatteryEnergyStatus[period] = "%.2f"%v.varValue
				
			elif "batterySOC" in v.name:
				period = int(v.name.split("_")[1])
				self.__BatterySOC[period] = "%.2f"%v.varValue
								
			elif "batteryCDStatus" in v.name:
				period = int(v.name.split("_")[1])
				self.__BatteryCDStatus[period] = "%d"%v.varValue
				
			else:
				pass
		

if __name__ == '__main__':
	main()