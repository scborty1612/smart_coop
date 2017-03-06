# Import stuffs


from agents.HomeAgent import HomeAgent
from agents.PredictionAgent import PredictionAgent

import aiomas
from sqlalchemy import create_engine

import sys, traceback
import datetime

def main():
	
	# Initiate the container
	C = aiomas.Container.create(('localhost', 5555))

	# List of valid houses
	# The houses are from the pecan street data
	agentIDs = [9019,  7881, 100237, 7850, 980, 9981,]

	# Create the DB engine
	db_engine = create_engine("mysql+pymysql://root@127.0.0.1/open_database")

	# Starting the Service Handler Agent
	# The purpose of this agent is to register the currently running agents
	# and on-request provides the instances of the agent
	

	# Now, initiate the actual agents with the houses
	homeAgents = [HomeAgent(container=C, agent_id=agent_id, db_engine=db_engine) for agent_id in agentIDs]
	for homeAgent in homeAgents:
		print(homeAgent.agent_id)

	# Creating and initiating a prediction agent
	predictionAgent = PredictionAgent(container=C, agent_id=1)
	# print(predictionAgent.addr)

	# Set up an experiment date
	# for which, we will try to perform agents' performances
	experiment_datetime = "2015-01-15"

	print("Experiment date is {}".format(experiment_datetime))

	# TEST
	#
	try:
		# Probably, running seperate event loop is not a good idea
		# but can do it for the time being :)
		for homeAgent in homeAgents:
			# Collect the historical data
			aiomas.run(until=predictionAgent.collectHistoricalData(agent=homeAgent))

		# Just try to get a fake prediction for an agent
		for homeAgent in homeAgents:
			aiomas.run(until=homeAgent.getLoadPrediction(prediction_agent=predictionAgent, experiment_datetime=experiment_datetime))


	except Exception as e:
		traceback.print_exc(file=sys.stdout)

	# Shutting down
	print("Shutting down the agents' platform!")
	C.shutdown()	

if __name__ == "__main__":
	main()