python mnist_sparce.py --sim_id 1 --num_epochs 100 --batch_size 20 --dim_res 1000 --perc_n 50.0 --ff_scale 0.3 --seed 42 &
python mnist_sparce.py --sim_id 2 --num_epochs 100 --batch_size 20 --dim_res 1000 --perc_n 75.0 --ff_scale 0.3 --seed 42 &
python mnist_sparce.py --sim_id 3 --num_epochs 100 --batch_size 20 --dim_res 1000 --perc_n 25.0 --ff_scale 0.3 --seed 42

wait

python mnist_sparce.py --sim_id 4 --num_epochs 100 --batch_size 20 --dim_res 2000 --perc_n 50.0 --ff_scale 0.3 --seed 42 &
python mnist_sparce.py --sim_id 5 --num_epochs 100 --batch_size 20 --dim_res 2000 --perc_n 75.0 --ff_scale 0.3 --seed 42 &
python mnist_sparce.py --sim_id 6 --num_epochs 100 --batch_size 20 --dim_res 2000 --perc_n 25.0 --ff_scale 0.3 --seed 42

wait

python mnist_sparce.py --sim_id 7 --num_epochs 100 --batch_size 20 --dim_res 1000 --perc_n 50.0 --ff_scale 0.4 --seed 42 &
python mnist_sparce.py --sim_id 8 --num_epochs 100 --batch_size 20 --dim_res 1000 --perc_n 75.0 --ff_scale 0.4 --seed 42 &
python mnist_sparce.py --sim_id 9 --num_epochs 100 --batch_size 20 --dim_res 1000 --perc_n 25.0 --ff_scale 0.4 --seed 42

wait

python mnist_sparce.py --sim_id 10 --num_epochs 100 --batch_size 20 --dim_res 2000 --perc_n 50.0 --ff_scale 0.4 --seed 42 &
python mnist_sparce.py --sim_id 11 --num_epochs 100 --batch_size 20 --dim_res 2000 --perc_n 75.0 --ff_scale 0.4 --seed 42 &
python mnist_sparce.py --sim_id 12 --num_epochs 100 --batch_size 20 --dim_res 2000 --perc_n 25.0 --ff_scale 0.4 --seed 42

wait

python mnist_sparce.py --sim_id 13 --num_epochs 100 --batch_size 20 --dim_res 1000 --perc_n 50.0 --ff_scale 0.5 --seed 42 &
python mnist_sparce.py --sim_id 14 --num_epochs 100 --batch_size 20 --dim_res 1000 --perc_n 75.0 --ff_scale 0.5 --seed 42 &
python mnist_sparce.py --sim_id 15 --num_epochs 100 --batch_size 20 --dim_res 1000 --perc_n 25.0 --ff_scale 0.5 --seed 42

wait

python mnist_sparce.py --sim_id 16 --num_epochs 100 --batch_size 20 --dim_res 2000 --perc_n 50.0 --ff_scale 0.5 --seed 42 &
python mnist_sparce.py --sim_id 17 --num_epochs 100 --batch_size 20 --dim_res 2000 --perc_n 75.0 --ff_scale 0.5 --seed 42 &
python mnist_sparce.py --sim_id 18 --num_epochs 100 --batch_size 20 --dim_res 2000 --perc_n 25.0 --ff_scale 0.5 --seed 42

wait
