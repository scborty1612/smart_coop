"""
Trigger the home agents in parallel
Make sure to run this after both root- and home-container are activated.
"""
# Import required packages
import subprocess
import os
import time
import multiprocessing

from util import HomeList as HL

def parallelRun():

	# List of home agents
	homes = HL.get()
	
	# Add the obvserver agent
	# agents.append(-3)
 
	# List of ports
	port_start = 5580
	ports = [port_start+i for i in range(len(homes))]

	# Set the command
	command = "python"

	# List of parallel processes
	processess = set()

	# Maximum number of processes
	max_processoes = multiprocessing.cpu_count()

	# Download the datasources
	for agent, port in zip(homes, ports):

		# Strining up the command line with appropriate agent id and command id
		processess.add(subprocess.Popen([command, "trigger_home.py", "--agent-id={}".format(agent), "--port={}".format(port), "--session-id=abcd1234"]))
		time.sleep(3)
		# if len(processess) >=max_processoes:
		# 	os.wait()
		# 	processess.difference_update([p for p in processess if p.poll() is not None])

if __name__ == '__main__':
	parallelRun()