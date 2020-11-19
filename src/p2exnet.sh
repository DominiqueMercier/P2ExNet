echo "Current working dir: $PWD"

DATASET="data/character_trajectories/dataset_steps-20_timesteps-206.pickle"

echo "=================================================="
echo "Start experiments: $DATASET"
echo "=================================================="

echo "Train blackbox model"
python3 src/main.py --data_path $DATASET --train --no_int

echo "Train interpretable model"
python3 src/main.py --data_path $DATASET --train --decode_latent_close --decode_latent

echo "Plot latent space comparison"
python3 src/main.py --data_path $DATASET --plot_latent

echo "Compute prototype quality"
python3 src/main.py --data_path $DATASET --quality

echo "Build evaluation mode"
python3 src/main.py --data_path $DATASET --evaluate --build_compare

echo "Evaluate mode"
python3 src/main.py --data_path $DATASET --evaluate

echo "=================================================="
echo "Finished"
echo "=================================================="