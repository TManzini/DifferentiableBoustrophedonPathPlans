import argparse
import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from shape_utils import recenter_centroid_at_origin, rescale_polygon
from boustrophedon.differentiable import generate_polygon_weights_and_biases_from_coordinates, get_transect_points, score_polygon_orientation, rotate_polygon

def cover_space(polygon_coords, transect_spacing, points_per_transect, polygon_temp, transect_falloff_temp, evaluation_intervals=1, polygon_rescale=1.0, maximize=False):
	#Center the coordinates of the polygon at the origin
	centered_coords = recenter_centroid_at_origin(polygon_coords)

	#Rescale them so the diameter of the polygon is 1.0 units
	scaled_coords, scale_factor = rescale_polygon(centered_coords, polygon_rescale)

	#Generate the matrix representations of the faces of the polygon
	line_components = generate_polygon_weights_and_biases_from_coordinates(scaled_coords)

	#Generate the transects that we are going to use to evaluate the polygon
	transects = get_transect_points(-polygon_rescale/2, polygon_rescale/2, -polygon_rescale/2, polygon_rescale/2, transect_spacing*scale_factor, points_per_transect)

	thetas = []
	scores = []
	gradients = []

	for angle in np.arange(0, 180, evaluation_intervals):

		#Start theta right in the middle so it can go in either direction...
		theta = torch.tensor([float(angle)], requires_grad=True)

		#Rotate the polygon
		rotated_line_weights, line_biases = rotate_polygon(line_components, theta)
		
		#Compute the Score
		score = score_polygon_orientation(transects, rotated_line_weights, line_biases, torch.tensor([0.0], requires_grad=True), polygon_temp, transect_falloff_temp, 0.5, 0.5)

		#Multiply by -1 so the optimizer will maximize instead of minimize
		if(maximize):
			maximize_score = score
		else:
			maximize_score = score * -1

		thetas.append(float(theta))
		scores.append(float(maximize_score))

	return thetas, scores

if(__name__ == "__main__"):

	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--transect_spacing', type=float, help='The spacing between transects at the scale of the passed polyon', default=0.5)
	parser.add_argument('--points_per_transect', type=int, help='The number of points that make up the approximation of the transect', default=1000)
	parser.add_argument('--evaluation_interval', type=float, help='The interval for angles at which we will evaluate the score', default=1.0)
	parser.add_argument('--polygon_rescale_term', type=float, help='The scale at which the polygon will be resized to prior to optimization', default=1.0)
	parser.add_argument('--polygon_coordinates_json_path', type=str, help="The path to a file which contains the coordinates of the polygon to be inspected", default=None)
	parser.add_argument('--output_folder_path', type=str, help="The path to the folder where the output image should be saved.", default=".")
	args = parser.parse_args()

	coords = [
			[8,4],
			[8,10],
			[4,8],
			[4,2],
		]

	if(args.polygon_coordinates_json_path):
		coords = json.load(open(args.polygon_coordinates_json_path, "r"))

	ax1 = plt.subplot()
	ax1.set_ylabel("Score")

	lines = []

	for temperature in np.logspace(4, 7, 10, base=2):
		thetas, scores = cover_space(coords,
	                                 args.transect_spacing,
	                                 args.points_per_transect,
	                                 temperature,
	                                 temperature,
	                                 args.evaluation_interval,
	                                 args.polygon_rescale_term,
	                                 True)
		
		l, = ax1.plot(thetas, scores, label="T=" + str(np.around(temperature, decimals=3)))
		lines.append(l)
	plt.legend(lines, [l.get_label() for l in lines], loc="center left")

	plt.xlabel("Angle")
	plt.title("Optimization Space\nPolygon Temp: Varies | Transect Spacing: " + str(args.transect_spacing) + "\nPoints Per Transect: " + str(args.points_per_transect) + " | Evaluation Interval: " + str(args.evaluation_interval) + " | Polygon Rescale: " + str(args.polygon_rescale_term))
	plt.legend()
	plt.savefig(os.path.join(args.output_folder_path, "score_surface_varies_" + str(args.transect_spacing) + "_" + str(args.points_per_transect) + "_" + str(args.evaluation_interval) + str(args.polygon_rescale_term) + ".png"))
	plt.clf()