import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from alive_progress import alive_bar
from polygenerator import random_convex_polygon

from shape_utils import recenter_centroid_at_origin, rescale_polygon
from boustrophedon.differentiable import generate_polygon_weights_and_biases_from_coordinates, get_transect_points, score_polygon_orientation, rotate_polygon
from boustrophedon.discrete import get_transects, rotate_polygon_discrete, discrete_score_polygon_orientation, make_polygon_from_points

def compare_discrete_v_gradient_settings(polygon, transect_spacing, theta, x_offset, points_per_transect, polygon_temp, transect_falloff_temp, polygon_rescale):
	#Center the coordinates of the polygon at the origin
	centered_coords = recenter_centroid_at_origin(polygon_coords)

	#Rescale them so the diameter of the polygon is 1.0 units
	scaled_coords, scale_factor = rescale_polygon(centered_coords, polygon_rescale)

	#Generate the matrix representations of the faces of the polygon
	line_components = generate_polygon_weights_and_biases_from_coordinates(scaled_coords)
	discrete_polygon = make_polygon_from_points(scaled_coords)

	#Generate the transects that we are going to use to evaluate the polygon
	gradient_transects = get_transect_points(-polygon_rescale/2, polygon_rescale/2, -polygon_rescale/2, polygon_rescale/2, transect_spacing*scale_factor, points_per_transect)
	discrete_transects = get_transects(-polygon_rescale/2, polygon_rescale/2, -polygon_rescale/2, polygon_rescale/2, transect_spacing*scale_factor)

	#Rotate the polygon
	rotated_line_weights_gradient, line_biases_gradient = rotate_polygon(line_components, torch.tensor([theta], requires_grad=True))
	rotated_polygon = rotate_polygon_discrete(discrete_polygon, theta)

	gradient_score = score_polygon_orientation(gradient_transects, rotated_line_weights_gradient, line_biases_gradient, torch.tensor([x_offset*scale_factor]), polygon_temp, transect_falloff_temp)
	discrete_score = discrete_score_polygon_orientation(discrete_transects, rotated_polygon, x_offset*scale_factor)

	return float(gradient_score), float(discrete_score)

num_polygons = 1000
polygon_rescale = 1.0

theta_random = np.random.default_rng()
x_offset_random = np.random.default_rng()
transect_spacing_random = np.random.default_rng()
polygon_point_random = np.random.default_rng()

points_per_transect_range = [100, 1000, 10000]
temperature_range = [1, 10, 100, 1000, 10000]

min_transect_spacing = 10e-4

errors = {k:{} for k in points_per_transect_range}
for k in errors.keys():
	errors[k] = {j:[] for j in temperature_range}

with alive_bar(len(points_per_transect_range) * len(temperature_range) * num_polygons) as bar:
	for temp in temperature_range:
		for points_per_transect in points_per_transect_range:
			n = 0
			while n < num_polygons:
				try:
					polygon_coords = random_convex_polygon(num_points=polygon_point_random.integers(3, 10))
					polygon_coords, _ = rescale_polygon(polygon_coords, polygon_rescale)

					transect_spacing = 0.5#transect_spacing_random.random()/3 #Divide by 3 so that there will be at minimum 3 transects 

					if(transect_spacing < min_transect_spacing):
						raise Exception("Transect spacing is too small.")

					theta = theta_random.random()*360 #Multiply by 360 in order to get a unique degree
					x_offset = x_offset_random.random()*transect_spacing - (transect_spacing/2) #Math to make it so that we will uniformly randomly sample the space between transects

					g, d = compare_discrete_v_gradient_settings(polygon_coords, transect_spacing, theta, x_offset, points_per_transect, temp, temp, polygon_rescale)

					if np.isnan(g) or np.isnan(d):
						raise Exception("Got NaN")

					errors[points_per_transect][temp].append(np.absolute(g-d))
					n += 1
					bar()
				except Exception:
					pass

for k in errors.keys():
	for j in errors[k]:
		print("Average error for", k, "points per transect and temperature of", j, "was:", np.mean(errors[k][j]), " (N=" + str(len(errors[k][j])) + ")")
	print("\n\n")
