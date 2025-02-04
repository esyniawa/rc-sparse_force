python generate_movement_dataset.py --save_dir arm_data --num_t 130 --num_init_thetas 1000 --num_goals 1000 --wait_steps_after_trajectory 20 --movement_duration 5.0 --train_split 0.8

python movement_sparce.py --sim_id 1 --num_epochs 50 --batch_size 32 --dim_res 1000 --perc_n 50.0 --save_dir arm_data --seed 42 &
python movement_sparce.py --sim_id 2 --num_epochs 50 --batch_size 32 --dim_res 1000 --perc_n 75.0 --save_dir arm_data --seed 42 &
python movement_sparce.py --sim_id 3 --num_epochs 50 --batch_size 32 --dim_res 1000 --perc_n 25.0 --save_dir arm_data --seed 42 &
python movement_sparce.py --sim_id 4 --num_epochs 50 --batch_size 32 --dim_res 1000 --perc_n 90.0 --save_dir arm_data --seed 42

wait

