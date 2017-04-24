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

# Plotting and stuffs
import matplotlib.pyplot as plt
import seaborn as sns

# Logging stuffs
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt


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

		# _, ax = plt.subplots()
		# plt.plot(self.predicted_demand, label="Demand P")
		# plt.plot(self.predicted_gen, label="Demand G")
		# plt.legend()
		# plt.show()


	def optimize(self, ):

		#d = 40
		# Constant degradation factor for the battery
		degrad_battery = 0.02
		
		# Epson to replace zero
		epson = 1e-7
		
		# Total daily load (that will not be changed)
		# The optimizer will not try to minimize the total demand,
		# instead schedule the battery that will minimize the load factor (peak to average ratio)
		total_demand = np.sum(self.revised_demand)

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
			# batteryProb += (Pb[t]  == self.predicted_demand[t] - self.predicted_gen[t]), "power_balance_at_period_%d"%t
			
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
		
		# Going for solving the optimization problem
		batteryProb.solve(COIN(msg = 1, threads=4))
		
		# Check the status (anything except 'optimal' is not expected)
		logging.info(pulp.LpStatus[batteryProb.status])
		
		# Record the total cost (approximated)
		self.__ObjectiveValue = value(batteryProb.objective)
		
		# Record the decision variables/status/parameters
		batteryEnergyStatus = np.zeros(self.periods + 1)        
		batterySOC = np.zeros(self.periods + 1)
		batteryCDStatus = np.zeros(self.periods + 1)
		batteryPower = np.zeros(self.periods + 1)
		
		# Record the variables with values
		for v in batteryProb.variables():             
			if  "batteryDispatch" in v.name:
				period = int(v.name.split("_")[1])
				batteryPower[period] = "%.2f"%v.varValue
				
			if  "batteryEnergyStatus" in v.name:
				period = int(v.name.split("_")[1])
				batteryEnergyStatus[period] = "%.2f"%v.varValue
				
			elif "batterySOC" in v.name:
				period = int(v.name.split("_")[1])
				batterySOC[period] = "%.2f"%v.varValue
								
			elif "batteryCDStatus" in v.name:
				period = int(v.name.split("_")[1])
				batteryCDStatus[period] = "%d"%v.varValue
				
			else:
				pass
		
		return batteryPower, batteryEnergyStatus


class SchedulerCombined(object):
	"""
	Class for multi-battery scehduler distributed over 
	different agents.
	"""

	def __init__(self, agent_info=None, granular=15, periods=None, ):

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
		
		# Agent information 
		self.__agent_info = agent_info

		self.__processAgentInfo()

	def __processAgentInfo(self, ):
		"""
		Make the agent information to be able to perform the optimization
		"""

		# 
		self.__agents_w_battery = list()
		self.__battery_information = list()

		self.__revised_load = np.zeros(self.periods+1)

		# Revised load \sum(predicted demand-predicted pv)

		for agent_id, agent_info in self.__agent_info.items():
			try:
				if agent_info['battery_info'] != 'NO BATTERY':
					self.__battery_information.append(agent_info['battery_info'])
					self.__agents_w_battery.append(agent_id)

				demand_pred = np.array(agent_info['demand_p']['load_prediction'])
				gen_pred = np.array(agent_info['gen_p']['pv_prediction'])
				
				this_revised_load = np.zeros(self.periods+1)
				this_revised_load[1:] = demand_pred - gen_pred
				self.__revised_load += this_revised_load
			except Exception as e:
				logging.info("Error in {}".format(agent_id))
				import traceback
				traceback.print_exc(file=sys.stdout)

		# _, ax = plt.subplots()
		# plt.plot(self.__revised_load)
		# plt.show()
		# plt.close()

	def optimize(self, ):

		# Constant degradation of battery
		degrad_battery = 0.02
		
		# Epson to replace zero
		epson = 1e-7
		
		# Total daily load (that will not be changed)
		# The optimizer will not try to minimize the total demand,
		# instead schedule the battery that will minimize the load factor (peak to average ratio)
		total_demand = np.sum(self.__revised_load)

		# Total number of batteries
		batteryNo = len(self.__battery_information)

		# Derive the linear programming based formulation
		batteryProb = LpProblem("BatteryScheduling", LpMinimize)
		
		# time index
		tIn = [i for i in range(self.periods+1)]

		# Battery Index    
		bIn = [k for k in range(batteryNo)]
		
		# Define the decision variables
		
		# Dispatched battery power [Real variable; continuous]
		Pb = pulp.LpVariable.dicts("batteryDispatch", (bIn, tIn))
		
		# Battery energy status []
		Xb = pulp.LpVariable.dicts("batteryEnergyStatus", (bIn, tIn), lowBound = 0)
		
		# Battery status variable (1 charge, 0 discharge) [Binary]
		Sb = pulp.LpVariable.dicts("batteryChargeDischargeStatus", (bIn, tIn), cat=pulp.LpBinary)
		 
		# Auxiliary variables to handle logical (in)equivalences
		Ax = pulp.LpVariable.dicts("auxEnergyStatus", (bIn, tIn), cat=pulp.LpContinuous)
		
		# Handling the max(k)
		J = pulp.LpVariable("J", cat=pulp.LpContinuous)

		
		# Objective function [Minimizing Periods*J]        
		batteryProb +=  self.periods * J
		
		
		"""Constraints and model definition"""

		for k,B in enumerate(self.__battery_information):
			# initialize the period 0
			# print(B)
			batteryProb += (Xb[k][0] == B['capacity'] * B['soc']), "initialize_b_state_%d0"%k
				
			# batteryProb += (Sb[k][0] == 0), "initialize_cd_state_%d0"%k


		# For each, have the battery constraints ready
		for t in range(self.periods+1)[1:]:
			# Constraint in J
			batteryProb += (J >= self.__revised_load[t] + sum(Pb[k][t] for k in range(len(self.__battery_information)))), "max_J_at_%d"%t
			
			for k,B in enumerate(self.__battery_information):						
				# Battery operation (changes in SOC by charging or discharging the battery with corresponding efficiencies)
				batteryProb += (Xb[k][t] == Xb[k][t-1] + \
					((B['c_eff']  - 1/B['d_eff']) * Ax[k][t] + \
						1/B['d_eff'] * Pb[k][t]) * self.__delT - degrad_battery), "battery_soc_change_at_period_%d_%d"%(k,t)
					
				# The battery energy level should be within the SOC limits
				batteryProb += (Xb[k][t] <= B['capacity'] * B['soc_high']),"battery_energy_up_at_period_%d_%d"%(k,t)
				batteryProb += (Xb[k][t] >= B['capacity'] * B['soc_low'] ),"battery_energy_down_at_period_%d_%d"%(k,t)
				
				# Followings are additional MIP constraints to handle inequalities in SOC dynamics

				# Discharge 
				batteryProb += (   B['d_rating']             * Sb[k][t]  + ( 0)* Ax[k][t]  - ( 1) * Pb[k][t] - B['d_rating'] <= 0)
				batteryProb += (- (B['d_rating'] + epson)    * Sb[k][t]  + ( 0)* Ax[k][t]  - (-1) * Pb[k][t] - (-epson)      <= 0)
				batteryProb += (   B['d_rating']             * Sb[k][t]  + ( 1)* Ax[k][t]  - ( 1) * Pb[k][t] - B['d_rating'] <= 0)
				batteryProb += (   B['d_rating']             * Sb[k][t]  + (-1)* Ax[k][t]  - (-1) * Pb[k][t] - B['d_rating'] <= 0)

				# Charge
				batteryProb += ( Ax[k][t]  <= B['c_rating']  * Sb[k][t])
				batteryProb += (- Ax[k][t] <= B['c_rating']  * Sb[k][t])
		
				
		# total load should not change
		batteryProb += (sum(self.__revised_load[t]+sum(Pb[k][t] for k in range(len(self.__battery_information))) \
							for t in range(self.periods+1)[1:]) == total_demand), "total_load_to_be_same"

		# Solve the battery problem using COIN-R 
		batteryProb.writeLP("problem_model_combined.lp")
		
		# Going for solving the optimization problem
		# Allow 20 mins of CPU time to perform the optimization
		# if still not yeild result, provide the current best
		batteryProb.solve(COIN(msg = 1, threads=8, maxSeconds=1200))
		
		# Check the status (anything except 'optimal' is not expected)
		logging.info(pulp.LpStatus[batteryProb.status])

		# Record the total cost (approximated)
		self.__ObjectiveValue = value(batteryProb.objective)
		
		# Record the decision variables/status/parameters
		self.batteryDispatch     = np.zeros((batteryNo) * (self.periods+1)).reshape(batteryNo, self.periods+1)
		self.batteryEnergyStatus = np.zeros((batteryNo) * (self.periods+1)).reshape(batteryNo, self.periods+1)
		self.batteryStatus       = np.zeros((batteryNo) * (self.periods+1)).reshape(batteryNo, self.periods+1)		
		
		for v in batteryProb.variables():
			vName = v.name.split("_")
			
			if "batteryDispatch" in v.name:
				#print v.name
				#print v.varValue
				bId = int(vName[1])
				tId = int(vName[2])
				self.batteryDispatch[bId][tId] = "%.2f"%v.varValue 
	
			elif "batteryEnergyStatus" in v.name:
				bId = int(vName[1])
				tId = int(vName[2])
				self.batteryEnergyStatus[bId][tId] = v.varValue
	
			elif "batteryChargeDischargeStatus" in v.name :
				bId = int(vName[1])
				tId = int(vName[2])                        
				self.batteryStatus[bId][tId] = v.varValue            
		
		logging.info(self.batteryDispatch)
		logging.info(self.batteryStatus)

		# Bind the battery information

		return dict(zip(self.__agents_w_battery, self.batteryDispatch)), dict(zip(self.__agents_w_battery, self.batteryEnergyStatus))


	def plotCurrentResult(self, ):
		_, ax = plt.subplots(figsize=[16, 6])

		plt.plot(self.__revised_load[1:], label="Aggregated Revised Load (before battery)")
		p2a_before = np.max(self.__revised_load[1:])/np.mean(self.__revised_load[1:])

		aggregated_battery_power = np.zeros(self.periods + 1)

		for k, battery in enumerate(self.batteryDispatch):
			if k <= 0:
				plt.plot(self.batteryDispatch[k][1:], label="Battery power from/to {}".format(self.__agents_w_battery[k]))
				plt.plot(self.batteryEnergyStatus[k][1:], label="Battery Energy status {}".format(self.__agents_w_battery[k]))

			aggregated_battery_power[1:] += np.array(self.batteryDispatch[k][1:])

		revise_load_after_battery = self.__revised_load[1:]+aggregated_battery_power[1:]
		p2a_after = np.max(revise_load_after_battery)/np.mean(revise_load_after_battery)

		plt.plot(revise_load_after_battery, label="Aggregated Revised Load (after battery)")
		plt.title("Before: {}, After: {}".format(p2a_before, p2a_after))

		plt.legend()

		plt.show()

		plt.close()

