python mnist_sparce.py --sim_id 1 --num_epochs 50 --batch_size 32 --dim_res 1000 --perc_n 50.0 --seed 42 &
python mnist_sparce.py --sim_id 2 --num_epochs 50 --batch_size 32 --dim_res 1000 --perc_n 75.0 --seed 42 &
python mnist_sparce.py --sim_id 3 --num_epochs 50 --batch_size 32 --dim_res 1000 --perc_n 90.0 --seed 42

wait

python mnist_sparce.py --sim_id 4 --num_epochs 50 --batch_size 32 --dim_res 2000 --perc_n 50.0 --seed 42 &
python mnist_sparce.py --sim_id 5 --num_epochs 50 --batch_size 32 --dim_res 2000 --perc_n 75.0 --seed 42 &
python mnist_sparce.py --sim_id 6 --num_epochs 50 --batch_size 32 --dim_res 2000 --perc_n 90.0 --seed 42

wait

