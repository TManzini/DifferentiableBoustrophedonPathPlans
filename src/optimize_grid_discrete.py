import argparse
import math
import torch
import numpy as np

from shape_utils import recenter_centroid_at_origin, rescale_polygon
from boustrophedon.discrete import get_transects, rotate_polygon_discrete, discrete_score_polygon_orientation, make_polygon_from_points

def optimize_discrete(polygon_coords, 
                      transect_spacing,
                      grid_samples,
                      polygon_rescale=1.0):

	#Center the coordinates of the polygon at the origin
	centered_coords = recenter_centroid_at_origin(polygon_coords)

	#Rescale them so the diameter of the polygon is 1.0 units
	scaled_coords, scale_factor = rescale_polygon(centered_coords, polygon_rescale)

	#Generate the polygon
	discrete_polygon = make_polygon_from_points(scaled_coords)

	#Generate the transects that we are going to use to evaluate the polygon
	min_transect_distance = ((-polygon_rescale/2)+(-transect_spacing/2))*1.5 #Half the width of the polygon, and half the width of a transect, times a safety factor of 1.5
	max_transect_distance = ((polygon_rescale/2)+(transect_spacing/2))*1.5
	discrete_transects = get_transects(min_transect_distance, max_transect_distance, -polygon_rescale/2, polygon_rescale/2, transect_spacing*scale_factor)
	
	#Make sure we dont fall off the edge of the transect spacing
	min_x_offset_limit = -1*(transect_spacing/2)
	max_x_offset_limit = (transect_spacing/2)
	
	best_theta = None
	best_x_offset = None
	best_score = -10e100

	x_offset_samples = int(np.around(grid_samples**0.5))
	theta_samples = int(np.around(grid_samples**0.5))


	x_offset_step = (max_x_offset_limit-min_x_offset_limit)/x_offset_samples
	x_offset_values = np.arange(min_x_offset_limit, max_x_offset_limit, x_offset_step)

	theta_step = (180)/theta_samples
	theta_values = np.arange(0, 180, theta_step)

	for theta in theta_values:
		for x_offset in x_offset_values:
			rotated_polygon = rotate_polygon_discrete(discrete_polygon, theta)
			discrete_score = discrete_score_polygon_orientation(discrete_transects, rotated_polygon, x_offset)
			if(discrete_score > best_score):
				best_score = discrete_score
				best_theta = theta
				best_x_offset = x_offset

	return best_theta, best_x_offset, best_score

if(__name__ == "__main__"):

	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--transect_spacing', type=float, help='The spacing between transects at the scale of the passed polyon', default=0.5)
	parser.add_argument('--points_per_transect', type=int, help='The number of points that make up the approximation of the transect', default=100)
	parser.add_argument('--grid_samples', type=int, help='The number of samples that will be used to define the grid. This will be rounded to the nearest square.', default=1000)
	parser.add_argument('--polygon_rescale', type=float, help='The scale at which the polygon will be resized to prior to optimization', default=1.0)
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

	theta, x_offset, score = optimize_discrete(coords,
                                 args.transect_spacing,
                                 args.grid_samples,
                                 args.polygon_rescale)

	print("\n\nGrid Solution Found...")
	print("\tScore:", float(score))
	print("\tTheta:", theta)
	print("\tX Offset:", x_offset)