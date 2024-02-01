import numpy as np
import math
import torch
import shapely

import matplotlib.pyplot as plt

from shape_utils import degree_to_radians, dist_between, get_polygon_centroid_from_coordinates, \
						get_max_polygon_width_from_coordinates, recenter_centroid_at_origin, \
						rescale_polygon

X = 0
Y = 1

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def make_polygon_from_points(points):
	return shapely.Polygon(points)

def get_transects(x_min, x_max, y_min, y_max, spacing):
	transects = []
	x = np.arange(x_min, x_max, spacing)
	for x_i in x:
		transect = [(x_i, y_max), (x_i, y_min)]
		transects.append(transect)
	return transects

def rotate_polygon_discrete(polygon, angle, origin=(0,0)):
	angle = np.radians(angle+90)
	rotated_points = []
	for point in polygon.exterior.coords:
		ox, oy = origin
		px, py = point

		qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
		qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
		rotated_points.append((qx, qy))

	return shapely.Polygon(rotated_points)

def compute_transect_length(transect, polygon, x_offset):
	transect = [(transect[0][0]+x_offset, transect[0][1]), (transect[1][0]+x_offset, transect[1][1])]
	line = shapely.LineString(transect)
	intersected_line = polygon.intersection(line)
	return intersected_line, intersected_line.length	

def discrete_score_polygon_orientation(transects, polygon, x_offset, a=0.5, b=0.5):
	transect_lengths = []
	
	#Score all the points in the transects
	for transect in transects:
		visible_transect, l = compute_transect_length(transect, polygon, x_offset)
		transect_lengths.append(l)	

	transects_shown = [1 if t > 0 else 0 for t in transect_lengths]
	
	#Estimate the total number of transects shown
	number_of_transects = sum(transects_shown)

	#Estimate the average length of the things we estimate we showed
	avg_len = sum(transect_lengths)/number_of_transects

	#Compute the estimated standard deviation of the things we thing we showed
	sum_squared_diff = 0
	for shown, t_len in zip(transects_shown, transect_lengths):
		#Compute the difference between the average length and the current transect, count it if it was shown, and then square it
		sum_squared_diff += ((t_len-avg_len)*shown)**2
	
	avg_squared_diff = sum_squared_diff/number_of_transects
	std_len = avg_squared_diff**0.5

	m_i_prime = avg_len
	cov_i = 1-(std_len)

	result_score = a*m_i_prime + b*cov_i

	return result_score
