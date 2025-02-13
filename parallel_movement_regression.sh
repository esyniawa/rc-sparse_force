python generate_movement_dataset.py --save_dir smoothed_arm_data --num_t 210 --num_init_thetas 1500 --num_goals 1500 --wait_steps 10 --movement_duration 5.0 --train_split 0.5

python movement_sparce.py --sim_id 1 --num_epochs 30 --ff_scale 1.0 --alpha 0.1 --prop_rec 0.1 --batch_size 32 --num_train_episodes 200 --dim_res 1000 --perc_n 50.0 --data_set_dir smoothed_arm_data --seed 42 &
python movement_sparce.py --sim_id 2 --num_epochs 30 --ff_scale 1.0 --alpha 0.1 --prop_rec 0.1 --batch_size 32 --num_train_episodes 200 --dim_res 1000 --perc_n 75.0 --data_set_dir smoothed_arm_data --seed 42 &
python movement_sparce.py --sim_id 3 --num_epochs 30 --ff_scale 1.0 --alpha 0.1 --prop_rec 0.1 --batch_size 32 --num_train_episodes 200 --dim_res 1000 --perc_n 25.0 --data_set_dir smoothed_arm_data --seed 42 &
python movement_sparce.py --sim_id 4 --num_epochs 30 --ff_scale 1.0 --alpha 0.1 --prop_rec 0.1 --batch_size 32 --num_train_episodes 200 --dim_res 1000 --perc_n 90.0 --data_set_dir smoothed_arm_data --seed 42

wait

python movement_sparce.py --sim_id 5 --num_epochs 30 --ff_scale 1.0 --alpha 0.1 --prop_rec 0.1 --batch_size 32 --num_train_episodes 200 --dim_res 2000 --perc_n 50.0 --data_set_dir smoothed_arm_data --seed 42 &
python movement_sparce.py --sim_id 6 --num_epochs 30 --ff_scale 1.0 --alpha 0.1 --prop_rec 0.1 --batch_size 32 --num_train_episodes 200 --dim_res 2000 --perc_n 75.0 --data_set_dir smoothed_arm_data --seed 42 &
python movement_sparce.py --sim_id 7 --num_epochs 30 --ff_scale 1.0 --alpha 0.1 --prop_rec 0.1 --batch_size 32 --num_train_episodes 200 --dim_res 2000 --perc_n 25.0 --data_set_dir smoothed_arm_data --seed 42 &
python movement_sparce.py --sim_id 8 --num_epochs 30 --ff_scale 1.0 --alpha 0.1 --prop_rec 0.1 --batch_size 32 --num_train_episodes 200 --dim_res 2000 --perc_n 90.0 --data_set_dir smoothed_arm_data --seed 42

wait
