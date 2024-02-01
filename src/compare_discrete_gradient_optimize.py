import argparse
import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from alive_progress import alive_bar
from polygenerator import random_convex_polygon

from shape_utils import recenter_centroid_at_origin, rescale_polygon
from optimize_gd import optimize
from optimize_grid_discrete import optimize_discrete

def compare_discrete_v_gradient_optimize(polygon, transect_spacing, points_per_transect, start_theta=0, max_comparisons=200):
	_, _, gradient_score = optimize(polygon, transect_spacing, points_per_transect, start_theta=start_theta, max_epochs=max_comparisons, temperature_growth_rate=1.0)
	_, _, discrete_score = optimize_discrete(polygon, transect_spacing, grid_samples=max_comparisons)
	return float(gradient_score), float(discrete_score)

if(__name__ == "__main__"):
	parser = argparse.ArgumentParser()
	parser.add_argument('--points_per_transect', type=int, help='The number of points that make up the approximation of the transect', default=10000)
	parser.add_argument('--polygon_rescale_term', type=float, help='The scale at which the polygon will be resized to prior to optimization', default=1.0)
	parser.add_argument('--output_folder_path', type=str, help="The path to the folder where the output image should be saved.", default=".")
	parser.add_argument('--num_polygons', type=int, help='The number of polygons that should be used to compare grid search vs gradient descent', default=10)
	args = parser.parse_args()

	num_polygons = args.num_polygons
	polygon_rescale = args.polygon_rescale_term
	points_per_transect = args.points_per_transect

	transect_spacing_random = np.random.default_rng()
	polygon_point_random = np.random.default_rng()
	theta_random = np.random.default_rng()

	gradient_scores = []
	discrete_scores = []
	score_delta = []

	with alive_bar(num_polygons) as bar:
		n = 0
		while n < num_polygons:
			try:
				polygon_coords = random_convex_polygon(num_points=polygon_point_random.integers(3, 10))
				polygon_coords, _ = rescale_polygon(polygon_coords, polygon_rescale)

				transect_spacing = transect_spacing_random.random()/3 #Divide by 3 so that there will be at minimum 3 transects 
				
				g, d = compare_discrete_v_gradient_optimize(polygon_coords, transect_spacing, points_per_transect, start_theta=theta_random.random()*180)

				gradient_scores.append(g)
				discrete_scores.append(d)
				score_delta.append(g-d)
				n += 1
				bar()
			except ZeroDivisionError:
				pass

	info_str = "Score Delta, if positive then gradient returned higher scores, if negative grid returned higher scores."
	stat_str = "Average Score Delta:" + str(np.mean(score_delta))

	f = open(os.path.join(args.output_folder_path, "grid_vs_gradient_descent_comparison_results.txt"), "w")
	f.write(info_str + "\n")
	f.write(stat_str + "\n")
	f.close()

	print(info_str)
	print(stat_str)
