import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import argparse
import torch
import json
import os

from shape_utils import recenter_centroid_at_origin, rescale_polygon
from boustrophedon.differentiable import generate_polygon_weights_and_biases_from_coordinates, get_transect_points, score_polygon_orientation, rotate_polygon

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--transect_spacing', type=float, help='The spacing between transects at the scale of the passed polyon', default=0.75)
parser.add_argument('--points_per_transect', type=int, help='The number of points that make up the approximation of the transect', default=1000)
parser.add_argument('--polygon_temperature', type=float, help='The temperature term for the sigmoid that defines the edges of the polygon. (0=Flat, Inf=Sharp)', default=10000)
parser.add_argument('--transect_falloff_temperature', type=float, help='The temperature term for the sigmoid that if a polygon was shown or not. (0=Flat, Inf=Sharp)', default=10000)
parser.add_argument('--polygon_rescale_term', type=float, help='The scale at which the polygon will be resized to prior to optimization', default=1.0)
parser.add_argument('--polygon_coordinates_json_path', type=str, help="The path to a file which contains the coordinates of the polygon to be inspected", default=None)
parser.add_argument('--output_folder_path', type=str, help="The path to the folder where the output image should be saved.", default=".")
parser.add_argument('--theta_evaluation_interval', type=float, help="The interval at which theta is sampled to generate the optimization surface.", default=1.0)
parser.add_argument('--xoffset_evaluation_interval', type=float, help="The interval at which xoffset is sampled to generate the optimization surface.", default=0.00075)
args = parser.parse_args()

coords = [
        [8,4],
        [8,10],
        [4,8],
        [4,2],
    ]

if(args.polygon_coordinates_json_path):
    coords = json.load(open(args.polygon_coordinates_json_path, "r"))

points_per_transect = args.points_per_transect
polygon_temp = args.polygon_temperature
transect_falloff_temp = args.transect_falloff_temperature
polygon_rescale_term = args.polygon_rescale_term
                             
#Center the coordinates of the polygon at the origin
centered_coords = recenter_centroid_at_origin(coords)

#Rescale them so the diameter of the polygon is 1.0 units
scaled_coords, scale_factor = rescale_polygon(centered_coords, polygon_rescale_term)

transect_spacing = args.transect_spacing*scale_factor

#Generate the matrix representations of the faces of the polygon
line_components = generate_polygon_weights_and_biases_from_coordinates(scaled_coords)

#Generate the transects that we are going to use to evaluate the polygon
min_transect_distance = ((-polygon_rescale_term/2)+(-transect_spacing/2))*1.1 #Half the width of the polygon, and half the width of a transect, times a safety factor of 1.1
max_transect_distance = ((polygon_rescale_term/2)+(transect_spacing/2))*1.1
transects = get_transect_points(min_transect_distance, max_transect_distance, -polygon_rescale_term/2, polygon_rescale_term/2, transect_spacing, points_per_transect)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
theta = np.arange(0, 180, args.theta_evaluation_interval)
x_offset = np.arange(-transect_spacing/2, transect_spacing/2, args.xoffset_evaluation_interval)

total = len(theta)*len(x_offset)

thetas, x_offsets = np.meshgrid(theta, x_offset)
scores = []
i = 0
for ts, xos in zip(thetas, x_offsets):
    scores.append([])
    for t, xo in zip(ts, xos):
        rotated_line_weights, line_biases = rotate_polygon(line_components, torch.tensor([t]))
        score = score_polygon_orientation(transects, rotated_line_weights, line_biases, torch.tensor([xo]), polygon_temp, transect_falloff_temp)
        score = float(score)
        scores[-1].append(score)
        i += 1
scores = np.array(scores)

# Plot the surface.
surf = ax.plot_surface(thetas, x_offsets, scores, cmap=cm.coolwarm, rstride=1, cstride=1, edgecolor=None)

# Customize the z axis.
ax.set_zlim(0.4, 1)
ax.set_zticks([0.5,1.0])
ax.set_xticks([0, 45, 90, 135, 180])
ax.set_xlabel("Angle")
ax.set_ylabel("X Offset")
ax.set_zlabel("Score")

ax.view_init(30, 135)
fig.savefig(os.path.join(args.output_folder_path, "surface_angle_xoffset_2dspace.png"), dpi=400)