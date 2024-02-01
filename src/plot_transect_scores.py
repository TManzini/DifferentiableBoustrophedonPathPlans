import os
import argparse
import torch
import json
import numpy as np
import matplotlib.pyplot as plt

from shape_utils import recenter_centroid_at_origin, rescale_polygon
from boustrophedon.differentiable import generate_polygon_weights_and_biases_from_coordinates, get_transect_points, approximate_transect_length, rotate_polygon

def plot_transects(polygon_coords, transect_spacing, points_per_transect, polygon_temp, transect_falloff_temp, output_folder, angle=0, polygon_rescale=1.0, maximize=False):
	#Center the coordinates of the polygon at the origin
	centered_coords = recenter_centroid_at_origin(polygon_coords)

	#Rescale them so the diameter of the polygon is 1.0 units
	scaled_coords, scale_factor = rescale_polygon(centered_coords, polygon_rescale)

	#Generate the matrix representations of the faces of the polygon
	line_components = generate_polygon_weights_and_biases_from_coordinates(scaled_coords)

	#Generate the transects that we are going to use to evaluate the polygon
	gradient_transects = get_transect_points(-polygon_rescale/2, polygon_rescale/2, -polygon_rescale/2, polygon_rescale/2, transect_spacing*scale_factor, points_per_transect)

	#Start theta right in the middle so it can go in either direction...
	theta = torch.tensor([float(angle)], requires_grad=True)

	#Rotate the polygon
	rotated_line_weights, line_biases = rotate_polygon(line_components, theta)

	x_offset = 0.0
	
	proportions_of_transects_shown = []
	
	#Score all the points in the transects
	plt.clf()
	ax = plt.figure().add_subplot(projection='3d')
	for transect in gradient_transects:
		visible_transect, l = approximate_transect_length(transect, rotated_line_weights, line_biases, polygon_temp, torch.tensor([x_offset], requires_grad=True))
		proportions_of_transects_shown.append(l)

		x, y = zip(*transect.tolist())
		z = visible_transect.tolist()

		for i in range(len(z)-1):
			ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=plt.cm.copper(z[i]))
	ax.view_init(20, 48)
	ax.set_zticks([0, 1])
	ax.set_xticks([-0.5, 0, 0.5])
	ax.set_yticks([-0.5, 0, 0.5])
	ax.set_zlim(0, 2)
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Polygon Presence")
	plt.savefig(os.path.join(output_folder, "Transect_Visualization_" + str(polygon_temp) + "_" + str(transect_falloff_temp) + "_" + str(angle) + ".png"), dpi=400)


if(__name__ == "__main__"):

	parser = argparse.ArgumentParser()
	parser.add_argument('--transect_spacing', type=float, help='The spacing between transects at the scale of the passed polyon', default=0.5)
	parser.add_argument('--points_per_transect', type=int, help='The number of points that make up the approximation of the transect', default=10000)
	parser.add_argument('--polygon_temperature', type=float, help='The temperature term for the sigmoid that defines the edges of the polygon', default=100)
	parser.add_argument('--transect_falloff_temperature', type=float, help='The temperature term for the sigmoid that if a polygon was shown or not', default=100)
	parser.add_argument('--evaluation_theta', type=float, help='The angles at which we will evaluate the polygon', default=0.25)
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

	plot_transects(coords, args.transect_spacing, args.points_per_transect, args.polygon_temperature, args.transect_falloff_temperature, args.output_folder_path, args.evaluation_theta, args.polygon_rescale_term, True)
	
	