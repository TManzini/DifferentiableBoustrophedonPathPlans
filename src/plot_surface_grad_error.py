import os
import argparse
import torch
import json
import numpy as np
import matplotlib.pyplot as plt

from shape_utils import recenter_centroid_at_origin, rescale_polygon
from boustrophedon.differentiable import generate_polygon_weights_and_biases_from_coordinates, get_transect_points, score_polygon_orientation, rotate_polygon
from boustrophedon.discrete import get_transects, rotate_polygon_discrete, discrete_score_polygon_orientation, make_polygon_from_points

def cover_space(polygon_coords, transect_spacing, points_per_transect, polygon_temp, transect_falloff_temp, evaluation_intervals=1, polygon_rescale=1.0, maximize=False):
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

	thetas = []
	gd_scores = []
	discrete_scores = []
	gradients = []

	for angle in np.arange(0, 180, evaluation_intervals):
		#Start theta right in the middle so it can go in either direction...
		theta = torch.tensor([float(angle)], requires_grad=True)

		optimizer = torch.optim.SGD([theta], lr=50.0)

		#Rotate the polygon
		rotated_line_weights, line_biases = rotate_polygon(line_components, theta)
		rotated_polygon = rotate_polygon_discrete(discrete_polygon, float(theta))

		x_offset = 0.0
		
		#Compute the Score
		gd_score = score_polygon_orientation(gradient_transects, rotated_line_weights, line_biases, torch.tensor([x_offset], requires_grad=True), polygon_temp, transect_falloff_temp)
		discrete_score = discrete_score_polygon_orientation(discrete_transects, rotated_polygon, x_offset*scale_factor)

		#Multiply by -1 so the optimizer will maximize instead of minimize
		if(maximize):
			maximize_gd_score = gd_score
			maximize_discrete_score = discrete_score
		else:
			maximize_gd_score = gd_score * -1
			maximize_discrete_score = discrete_score * -1

		#Compute the gradient
		maximize_gd_score.backward()

		thetas.append(float(theta))
		gd_scores.append(float(maximize_gd_score))
		discrete_scores.append(float(maximize_discrete_score))
		gradients.append(float(theta.grad))

		optimizer.zero_grad()

	return thetas, gd_scores, discrete_scores, gradients

if(__name__ == "__main__"):

	parser = argparse.ArgumentParser()
	parser.add_argument('--transect_spacing', type=float, help='The spacing between transects at the scale of the passed polyon', default=0.5)
	parser.add_argument('--points_per_transect', type=int, help='The number of points that make up the approximation of the transect', default=10000)
	parser.add_argument('--polygon_temperature', type=float, help='The temperature term for the sigmoid that defines the edges of the polygon', default=100)
	parser.add_argument('--transect_falloff_temperature', type=float, help='The temperature term for the sigmoid that if a polygon was shown or not', default=100)
	parser.add_argument('--evaluation_interval', type=float, help='The interval for angles at which we will evaluate the score and gradient', default=0.25)
	parser.add_argument('--polygon_rescale_term', type=float, help='The scale at which the polygon will be resized to prior to optimization', default=1.0)
	parser.add_argument('--plot_approximation_error', action="store_true", help="When set, the generated plot will contain the approximation error.")
	parser.add_argument('--plot_gradients', action="store_true", help="When set, the generated plot will contain the gradients.")
	parser.add_argument('--output_folder_path', type=str, help="The path to the folder where the output image should be saved.", default=".")
	parser.add_argument('--polygon_coordinates_json_path', type=str, help="The path to a file which contains the coordinates of the polygon to be inspected", default=None)
	args = parser.parse_args()

	coords = [
			[8,4],
			[8,10],
			[4,8],
			[4,2],
		]

	if(args.polygon_coordinates_json_path):
		coords = json.load(open(args.polygon_coordinates_json_path, "r"))

	thetas, gd_scores, discrete_scores, gradients = cover_space(coords,
	                                        args.transect_spacing,
	                                        args.points_per_transect,
	                                        args.polygon_temperature,
	                                        args.transect_falloff_temperature,
	                                        args.evaluation_interval,
	                                        args.polygon_rescale_term,
	                                        True)
	
	ax1 = plt.subplot()
	ax1.set_ylabel("Score")
	ax1.set_xlabel("Angle")
	max_score = max(gd_scores)
	min_score = min(gd_scores)
	for i in range(0, len(thetas)-1):
		avg_score = (gd_scores[i] + gd_scores[i+1])/2
		norm = (avg_score-min_score)/(max_score-min_score)
		l1, = ax1.plot(thetas[i:i+2], gd_scores[i:i+2], color=plt.cm.coolwarm(norm))

	errors = np.array(gd_scores) - np.array(discrete_scores)

	if(args.plot_gradients):
		ax2 = ax1.twinx()
		ax2.set_ylabel("Gradient of Score")
		l2, = ax2.plot(thetas, gradients, color='green', label="Gradient of Score", alpha=0.4)
		ax2.legend()

	if(args.plot_approximation_error):
		ax2 = ax1.twinx()
		ax2.set_ylabel("Approximation Error")
		l2, = ax2.plot(thetas, errors, color='orange', label="Approximation Error", alpha=0.4)
		ax2.legend()

	plt.xlabel("Angle")
	plt.grid(alpha=0.2)
	plt.savefig(os.path.join(args.output_folder_path, "score_surface_" + str(args.polygon_temperature) + "_" + str(args.transect_falloff_temperature) + "_" + str(args.transect_spacing) + "_" + str(args.points_per_transect) + "_" + str(args.evaluation_interval) + str(args.polygon_rescale_term) + ".png"), dpi=400, bbox_inches='tight')
	plt.clf()