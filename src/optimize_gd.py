import argparse
import json
import math
import torch
import numpy as np

from shape_utils import recenter_centroid_at_origin, rescale_polygon
from boustrophedon.differentiable import generate_polygon_weights_and_biases_from_coordinates, get_transect_points, score_polygon_orientation, rotate_polygon, score_polygon_orientation_nonconvex
from boustrophedon.discrete import discrete_score_polygon_orientation, make_polygon_from_points, get_transects, rotate_polygon_discrete
from boustrophedon.closed_form import triangulated_face_groups, rotate_triangulated_faces

from plot_surface_grad_error import cover_space

def temperature_growth_function(temp, growth_rate, gradient, gradient_clip):
	max_growth_rate = growth_rate
	growth = growth_rate * math.fabs(1/(gradient*1000))
	if(growth > max_growth_rate):
		growth = max_growth_rate
	if(growth < 0.99):
		growth = 0.99
	return temp*growth

def optimize(polygon_coords, 
             transect_spacing,
             points_per_transect,
             sigmoid_temp=600,
             temperature_growth_rate=1.005,
             start_theta=180,
             start_x_offset=0.0,
             polygon_rescale=1.0,
             theta_gradient_clip=0.0005,
             theta_learning_rate=600,
             theta_momentum=0.8,
             x_offset_gradient_clip=0.25,
             x_offset_learning_rate=0.01,
             x_offset_momentum=0.5,
             early_stopping_gradient_value=10e-12,
             max_sigmoid_temp=600,
             top_k_retest=1,
             max_epochs=1000,
             use_closed_form=False,
             smooth_closed_form=False,
             nonconvex=False):

	#Center the coordinates of the polygon at the origin
	centered_coords = recenter_centroid_at_origin(polygon_coords)

	#Rescale them so the diameter of the polygon is 1.0 units
	scaled_coords, scale_factor = rescale_polygon(centered_coords, polygon_rescale)

	#nonconvex triangulates the (possibly concave) polygon once and treats each
	#triangle as its own tiny convex face set; per-transect length is the SUM of
	#the per-triangle lengths (exact, since triangles partition the polygon).
	#This only supports the closed-form path (hard or smoothed via
	#smooth_closed_form) -- there is no separate trapezoidal-only mode for
	#non-convex polygons here, since the closed form is already exact and faster.
	if nonconvex:
		face_groups = triangulated_face_groups(scaled_coords)
	else:
		#Generate the matrix representations of the faces of the polygon
		line_components = generate_polygon_weights_and_biases_from_coordinates(scaled_coords)

	#Generate the transects that we are going to use to evaluate the polygon
	min_transect_distance = ((-polygon_rescale/2)+(-transect_spacing/2))*1.5 #Half the width of the polygon, and half the width of a transect, times a safety factor of 1.5
	max_transect_distance = ((polygon_rescale/2)+(transect_spacing/2))*1.5
	transects = get_transect_points(min_transect_distance, max_transect_distance, -polygon_rescale/2, polygon_rescale/2, transect_spacing*scale_factor, points_per_transect)
	
	#Generate the discrete polygon and transects that will be used for evaluation at the end
	discrete_polygon = make_polygon_from_points(scaled_coords)
	discrete_transects = get_transects(min_transect_distance, max_transect_distance, -polygon_rescale/2, polygon_rescale/2, transect_spacing*scale_factor)

	#Make sure we dont fall off the edge of the transect spacing
	min_x_offset_limit = -1*(transect_spacing/2)
	max_x_offset_limit = (transect_spacing/2)

	#Start theta right in the middle so it can go in either direction...
	theta = torch.tensor([float(start_theta)], requires_grad=True)
	x_offset = torch.tensor([0.0], requires_grad=True)
	
	optimizer = torch.optim.SGD([{'params': [theta], "lr":theta_learning_rate, "momentum":theta_momentum}, 
		                         {'params': [x_offset], "lr":x_offset_learning_rate, "momentum":x_offset_momentum}])
	
	thetas = []
	x_offsets = []
	scores = []
	theta_gradients = []

	epoch = 0
	keep_going = True

	step_sigmoid_temp = sigmoid_temp
	
	while(epoch < max_epochs and keep_going):
		#print("Epoch:", epoch)

		#Compute the Score
		#use_closed_form replaces the trapezoidal length (Eq. 6) with the closed-form
		#chord length, needing no points-per-transect.
		#smooth_closed_form additionally reuses sigmoid_temp as the smoothness knob
		#"beta" of the soft chord (soft_closed_form_transect_length), so the same
		#anneal schedule that used to melt/solidify the trapezoidal integrand now
		#melts/solidifies the closed-form chord directly -- giving back the smooth
		#early landscape GD needs to escape local minima, without any integration.
		#use_closed_form alone (smooth_closed_form=False) is the exact hard chord:
		#temperature-free, maximum fidelity, but as kinked/non-convex as high Temp.
		closed_form_beta = sigmoid_temp if smooth_closed_form else None
		if nonconvex:
			#Rotate every triangle's faces with the same rigid transform.
			rotated_face_groups = rotate_triangulated_faces(face_groups, theta)
			score = score_polygon_orientation_nonconvex(transects, rotated_face_groups, x_offset, sigmoid_temp, sigmoid_temp, use_closed_form=True, closed_form_beta=closed_form_beta)
		else:
			#Rotate the polygon
			rotated_line_weights, line_biases = rotate_polygon(line_components, theta)
			score = score_polygon_orientation(transects, rotated_line_weights, line_biases, x_offset, sigmoid_temp, sigmoid_temp, use_closed_form=(use_closed_form or smooth_closed_form), closed_form_beta=closed_form_beta)

		#Multiply by -1 so the optimizer will maximize instead of minimize
		maximize_score = score * -1
		
		#Compute the gradient
		maximize_score.backward()

		thetas.append(float(theta)%360)
		x_offsets.append(float(x_offset))
		scores.append(float(maximize_score))
		theta_gradients.append(float(theta.grad))

		#Clip the theta_gradients so we dont explode out of the minima if we get there
		torch.nn.utils.clip_grad_norm_(theta, theta_gradient_clip)
		torch.nn.utils.clip_grad_norm_(x_offset, x_offset_gradient_clip)

		sigmoid_temp = temperature_growth_function(sigmoid_temp, temperature_growth_rate, float(theta.grad), theta_gradient_clip)
		sigmoid_temp = min(sigmoid_temp, max_sigmoid_temp)

		#Step the optimizer
		optimizer.step()

		if(x_offset < min_x_offset_limit):
			x_offset = torch.tensor([max_x_offset_limit], requires_grad=True) #wrap around
		if(x_offset > max_x_offset_limit):
			x_offset = torch.tensor([min_x_offset_limit], requires_grad=True) #wrap around

		#Reset the theta_gradients
		optimizer.zero_grad()

		#Update the epoch
		epoch += 1

		if(math.isnan(theta_gradients[-1]) or np.fabs(theta_gradients[-1]) < early_stopping_gradient_value):
			#print("Stopping early because gradient minimum value was reached.")
			keep_going = False

	#Get the top K for retesting now that the temperature has been raised.
	candidate_parameters = zip(scores, thetas, x_offsets)
	sorted_candidates = sorted(candidate_parameters)
	top_k_candidate_parameters = sorted_candidates[:top_k_retest]
	
	best_score = None
	best_theta = None
	best_x_offset = None
	for score, theta, x_offset in top_k_candidate_parameters:

		rotated_polygon = rotate_polygon_discrete(discrete_polygon, theta)
		n_score = discrete_score_polygon_orientation(discrete_transects, rotated_polygon, x_offset)
			
		if(best_score is None or n_score < best_score):
			best_score = n_score
			best_theta = theta
			best_x_offset = x_offset

	return best_theta, best_x_offset, best_score

if(__name__ == "__main__"):

	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--transect_spacing', type=float, help='The spacing between transects at the scale of the passed polyon', default=0.5)
	parser.add_argument('--points_per_transect', type=int, help='The number of points that make up the approximation of the transect', default=1000)
	parser.add_argument('--inital_sigmoid_temperature', type=float, help='The temperature term for all sigmoids. (0=Flat, Inf=Sharp)', default=10)
	parser.add_argument('--temperature_growth_rate', type=float, help='A term that is used to vary the temperature of the loss surface based on the gradient.', default=1.005)
	parser.add_argument('--polygon_rescale', type=float, help='The scale at which the polygon will be resized to prior to optimization', default=1.0)
	parser.add_argument('--start_theta', type=float, help='The scale at which the polygon will be resized to prior to optimization', default=180.0)
	parser.add_argument('--temp_max', type=float, help='The maximum temperature that will be used in the sigmoid functions.', default=600)
	parser.add_argument('--polygon_coordinates_json_path', type=str, help="The path to a file which contains the coordinates of the polygon to be inspected", default=None)
	parser.add_argument('--use_closed_form', action='store_true', help="Use the exact closed-form chord length instead of the trapezoidal approximation of Eq. 6 (temperature-free, maximum fidelity, kinked landscape).")
	parser.add_argument('--smooth_closed_form', action='store_true', help="Use the closed-form chord length with a smoothness knob (beta=sigmoid_temp) in place of the trapezoidal Temp, recovering a smooth early landscape without integration. Implies --use_closed_form.")
	parser.add_argument('--nonconvex', action='store_true', help="Triangulate the polygon (ear clipping) and sum per-triangle closed-form chord lengths, supporting concave polygons. Always uses the closed form (hard, or smoothed if --smooth_closed_form is also set).")
	args = parser.parse_args()

	coords = [
			[8,4],
			[8,10],
			[4,8],
			[4,2],
		]

	if(args.polygon_coordinates_json_path):
		coords = json.load(open(args.polygon_coordinates_json_path, "r"))

	theta, x_offset, score = optimize(coords,
                                 args.transect_spacing,
                                 args.points_per_transect,
                                 args.inital_sigmoid_temperature,
                                 args.temperature_growth_rate,
                                 args.start_theta,
                                 args.polygon_rescale,
                                 max_sigmoid_temp=args.temp_max,
                                 use_closed_form=args.use_closed_form,
                                 smooth_closed_form=args.smooth_closed_form,
                                 nonconvex=args.nonconvex)

	print("\n\nGD Solution converged...")
	print("\tScore:", float(score))
	print("\tTheta:", theta)
	print("\tX Offset:", x_offset)