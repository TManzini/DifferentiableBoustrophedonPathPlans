polygon_json_file="../data/parallelogram.json"
replication_output_folder="../replication_outputs"

echo "Generating a folder to save plots and other outputs at " $replication_output_folder
mkdir $replication_output_folder

echo "Generating 3d plots of transects intersecting a polygon at varying temperatures"
python plot_transect_scores.py --output_folder_path $replication_output_folder --transect_spacing 0.5 --points_per_transect 1000 --polygon_temperature 17 --transect_falloff_temperature 17 --evaluation_theta 0.1 --polygon_rescale_term 1.0 --polygon_coordinates_json_path $polygon_json_file
python plot_transect_scores.py --output_folder_path $replication_output_folder --transect_spacing 0.5 --points_per_transect 1000 --polygon_temperature 100 --transect_falloff_temperature 100 --evaluation_theta 0.1 --polygon_rescale_term 1.0 --polygon_coordinates_json_path $polygon_json_file

echo "Generating a 2d plot of the optimization surface at varying thetas at a fixed temperature and a fixed offset"
python plot_surface_grad_error.py --output_folder_path $replication_output_folder --transect_spacing 0.5 --points_per_transect 1000 --polygon_temperature 1000 --transect_falloff_temperature 1000 --evaluation_interval 0.1 --polygon_rescale_term 1.0 --polygon_coordinates_json_path $polygon_json_file
echo "Generating a 3d plot of the optimization surface at varying thetas, xoffset at a fixed temperature"
python plot_surface_theta_xoffset.py --output_folder_path $replication_output_folder --transect_spacing 0.5 --points_per_transect 1000 --polygon_rescale_term 1.0 --polygon_coordinates_json_path $polygon_json_file --theta_evaluation_interval 1.0 --xoffset_evaluation_interval 0.00075
echo "Generating a 3d plot of the optimization surface at varying thetas, temperatures at a fixed xoffset"
python plot_surface_theta_temperature.py --transect_spacing 0.5 --points_per_transect 1000 --theta_evaluation_interval 1.0 --temperature_evaluation_interval 5 --polygon_rescale_term 1.0 --polygon_coordinates_json_path $polygon_json_file --output_folder_path $replication_output_folder
echo "Measuring representation error between discrete and differentiable representations"
python compare_discrete_gradient.py --polygon_rescale_term 1.0 --num_polygons 1000 --output_folder_path $replication_output_folder
echo "Measuring average error between grid search and gradient descent... This will take a while..."
python compare_discrete_gradient_optimize.py --points_per_transect 1000 --polygon_rescale_term 1.0 --num_polygons 1000 --output_folder_path $replication_output_folder

echo "Generating a supplemental 2d plot of the optimization surface at varying thetas, temperatures at a fixed xoffset"
python plot_surface_temperature.py --output_folder_path $replication_output_folder --transect_spacing 0.5 --points_per_transect 1000 --evaluation_interval 1.0 --polygon_rescale_term 1.0 --polygon_coordinates_json_path $polygon_json_file
