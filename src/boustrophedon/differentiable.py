import numpy as np
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib.pyplot as plt

from shape_utils import degree_to_radians, dist_between, get_polygon_centroid_from_coordinates, \
						get_max_polygon_width_from_coordinates, recenter_centroid_at_origin, \
						rescale_polygon

X = 0
Y = 1

def get_transect_points(x_min, x_max, y_min, y_max, spacing, points_per_transect):
	transects = []
	x = np.arange(x_min, x_max, spacing)
	for x_i in x:
		y = np.arange(y_min, y_max, (y_max-y_min)/points_per_transect)
		transect_points = [(x_i, y_i) for y_i in y]
		transects.append(transect_points)
	return torch.tensor(transects, requires_grad=True)


def generate_line_between_points(x1,y1,x2,y2):
	
	m = np.array([
		[x1,y1,1],
		[x2,y2,1]
	])
	
	x_term = np.linalg.det(np.array([m[:, 0], m[:, 2]]).transpose())
	y_term = np.linalg.det(np.array([m[:, 1], m[:, 2]]).transpose())
	c_term = np.linalg.det(np.array([m[:, 0], m[:, 1]]).transpose())
	
	w = [x_term, y_term]
	return torch.tensor(w, requires_grad=True), c_term

def generate_polygon_weights_and_biases_from_coordinates(coordinates):
	line_components = []
	for i in range(0, len(coordinates)):
		line_components.append(generate_line_between_points(coordinates[i][X],
														    coordinates[i][Y],
														    coordinates[(i+1)%len(coordinates)][X],
														    coordinates[(i+1)%len(coordinates)][Y]))
	return line_components

def rotate_polygon(line_components, angle):
	#Rotate the line components as they were passed in
	rotated_weights = []
	biases = []

	rad = degree_to_radians(angle.squeeze())
	rot_stack = torch.stack([torch.cos(rad), -1*torch.sin(rad), 
		                     torch.sin(rad), torch.cos(rad)]).reshape(2,2)
	
	for w, b in line_components:
		w_rot = torch.matmul(rot_stack.double(), w.double())
		rotated_weights.append(w_rot)
		biases.append(b)

	return rotated_weights, biases 

def approximate_transect_length(transect, line_weights, line_biases, temp, x_offset):
	offset_coord = torch.cat((x_offset, torch.tensor([0])), 0)
	offset_transect = transect.add(offset_coord)
	points = offset_transect.repeat(1, 1, len(line_weights)).reshape(len(offset_transect), len(line_weights), 2)
	biases = torch.tensor(line_biases)
	weights = torch.stack(line_weights)

	wtxs_sum = torch.multiply(weights, points)
	wtxs = torch.sum(wtxs_sum, dim=2)
	wtxbs = torch.add(wtxs, biases)
	wtxbs_temp = torch.multiply(wtxbs, temp)
	scores = torch.sigmoid(wtxbs_temp)

	score_prod = torch.prod(scores, dim=1)
	score_prod = torch.subtract(score_prod, 0.5)
	score_prod = torch.multiply(score_prod, temp)
	score_prod = torch.sigmoid(score_prod)
	transect_length_prod = torch.trapz(score_prod, dx=1.0/points.size(dim=0))

	return score_prod, transect_length_prod

def score_polygon_orientation(transects, line_weights, line_biases, x_offset, polygon_temp, transect_falloff_temp, a=0.5, b=0.5):
	proportions_of_transects_shown = []
	
	#Score all the points in the transects

	for transect in transects:
		visible_transect, l = approximate_transect_length(transect, line_weights, line_biases, polygon_temp, x_offset)
		proportions_of_transects_shown.append(l)
	
	#Estimate if which transects we showed using the sigmoid function
	proportions_of_transects_shown_tensor = torch.stack(proportions_of_transects_shown)

	proportions_of_transects_shown_temp = torch.multiply(proportions_of_transects_shown_tensor, transect_falloff_temp)
	estimated_transects_shown = (torch.sigmoid(proportions_of_transects_shown_temp)-0.5)/0.5
	
	#Estimate the total number of transects shown
	estimated_number_of_transects = torch.sum(estimated_transects_shown)

	#Estimate the average length of the things we estimate we showed
	e_avg_len = torch.sum(proportions_of_transects_shown_tensor)/estimated_number_of_transects

	#Compute the estimated standard deviation of the things we thing we showed
	sum_squared_diff = 0
	for shown, e_len in zip(estimated_transects_shown, proportions_of_transects_shown):
		#Compute the difference between the average length and the current transect, count it if it was shown, and then square it
		sum_squared_diff += ((e_len-e_avg_len)*shown)**2
	
	e_avg_squared_diff = sum_squared_diff/estimated_number_of_transects
	e_std_len = e_avg_squared_diff**0.5

	m_i_prime = e_avg_len
	cov_i = 1-(e_std_len)

	result_score = a*m_i_prime + b*cov_i

	return result_score
