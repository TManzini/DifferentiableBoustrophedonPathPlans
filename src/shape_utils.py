import numpy as np

X = 0
Y = 1

def degree_to_radians(degrees):
	return degrees*(np.pi/180)

def dist_between(x1,y1,x2,y2):
	return ((x1-x2)**2 + (y1-y2)**2)**0.5
	
def get_polygon_centroid_from_coordinates(coordinates):
	x_list = [coord[X] for coord in coordinates]
	y_list = [coord[Y] for coord in coordinates]
	x = sum(x_list) / len(coordinates)
	y = sum(y_list) / len(coordinates)
	return (x, y)

def get_max_polygon_width_from_coordinates(coordinates):
	dists = []
	for i in range(0, len(coordinates)):
		for j in range(i+1, len(coordinates)):
			x_1, y_1 = coordinates[i]
			x_2, y_2 = coordinates[j]
			dists.append(dist_between(x_1, y_1, x_2, y_2))
	return max(dists)

def recenter_centroid_at_origin(coordinates):
	centroid = get_polygon_centroid_from_coordinates(coordinates)
	result = []
	for x, y in coordinates:
		x_new = x - centroid[X]
		y_new = y - centroid[Y]
		result.append([x_new, y_new])
	return result

def rescale_polygon(coordinates, max_width):
	current_max_width = get_max_polygon_width_from_coordinates(coordinates)
	scale_factor = max_width/current_max_width
	rescaled = []
	for x,y in coordinates:
		rescaled.append([x*scale_factor, y*scale_factor])
	return rescaled, scale_factor