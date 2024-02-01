# Differentiable Boustrophedon Path Plans

This repository contains the source code associated with the ICRA'24 paper "[Differentiable Boustrophendon Path Plans](https://arxiv.org/abs/2309.09882)"

To replicate the results of this paper, you can use the [replicate_icra2024.sh](https://github.com/TManzini/DifferentiableBoustrophedonPathPlans/blob/main/src/replicate_icra2024.sh) script. More explicitly, you should run the following commands...

    git clone https://github.com/TManzini/DifferentiableBoustrophedonPathPlans.git
    cd ./DifferentiableBoustrophedonPathPlans
    pip install -r requirements.txt
    cd ./src
    ./replicate_icra2024.sh

This will run several scripts to generate plots and data files, and output content to the console containing the results and plots in the paper. These results will be saved in a folder titled `replication_outputs` at the base directory of the repo. Should you encounter any issues, please raise them on this repository or contact the authors.

Happy Path Planning!
