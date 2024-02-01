import argparse
import torch
import json
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

num_polygons = 10
polygon_rescale = 1.0
points_per_transect = 1000

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

print("Score Delta, if positive then gradient returned higher scores, if negative grid returned higher scores.")
print("Average Score Delta:", np.mean(score_delta))
