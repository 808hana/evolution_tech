#!/usr/bin/env python3

import benchmark_functions as bf
from bees_algorithm import BeesAlgorithm, ParallelBeesAlgorithm
import dill, pickle

MAX_ITERATION = 10000

'''
These are minimum looking functions
'''
# the BA will search for a MAXIMUM in a region with a certain center and within an hypecube of a certain edge
# for this reason botht he function and its opposite will be included.
# the optimisation itself is pefromed on the opposite, whilst the minimum check is performed on the regular function
def look_within(function, function_opposite, centre, edge=1e-6, parallel=True, strict=True):
	solution = None
	is_minimum=False
	while solution is None or not is_minimum:
		lb, ub = [x-edge/2 for x in centre], [x+edge/2 for x in centre]
		alg = ParallelBeesAlgorithm(score_function=function_opposite,range_min=lb, range_max=ub)
		# this is a workaround to include the centre in the initial sites pop
		alg.current_sites[-1].centre = centre
		alg.current_sites[-1].score = function_opposite(centre)
		alg.performFullOptimisation(max_iteration=MAX_ITERATION)
		solution = alg.best_solution
		print(f"Best solution found is {solution.values} of score {solution.score} and function value {function(solution.values)}")
		testing = function.testLocalMinimum(solution.values, strict=strict)
		print(testing)
		is_minimum=testing[0]
		if not is_minimum:
			# replace the centre and keep looking
			print(f"> changing centre from {centre} to {testing[1]}")
			centre = testing[1]
	return solution

if __name__=='__main__':
	c=bf.EggHolder
	f = c(2)
	f_o = c(2,opposite=True) 
	r = (f.suggested_bounds()[1][0]-f.suggested_bounds()[0][0])*1e-7
	centre = [511.9999999689328, 404.23180491916895]
	s = look_within(f, f_o, centre, r, strict=False)
	print(s)