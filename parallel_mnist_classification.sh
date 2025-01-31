python mnist_sparce.py --num_epochs 50 --batch_size 16 --dim_res 1000 --perc_n 50.0 --prop_rec 0.1 --seed 42 &
python mnist_sparce.py --num_epochs 50 --batch_size 16 --dim_res 1000 --perc_n 75.0 --prop_rec 0.1 --seed 42 &
python mnist_sparce.py --num_epochs 50 --batch_size 16 --dim_res 1000 --perc_n 90.0 --prop_rec 0.1 --seed 42

wait

python mnist_sparce.py --num_epochs 50 --batch_size 16 --dim_res 1000 --perc_n 50.0 --prop_rec 0.2 --seed 42 &
python mnist_sparce.py --num_epochs 50 --batch_size 16 --dim_res 1000 --perc_n 75.0 --prop_rec 0.2 --seed 42 &
python mnist_sparce.py --num_epochs 50 --batch_size 16 --dim_res 1000 --perc_n 90.0 --prop_rec 0.2 --seed 42

wait

python mnist_sparce.py --num_epochs 50 --batch_size 16 --dim_res 1000 --perc_n 50.0 --prop_rec 0.5 --seed 42 &
python mnist_sparce.py --num_epochs 50 --batch_size 16 --dim_res 1000 --perc_n 75.0 --prop_rec 0.5 --seed 42 &
python mnist_sparce.py --num_epochs 50 --batch_size 16 --dim_res 1000 --perc_n 90.0 --prop_rec 0.5 --seed 42

# bigger reservoir
python mnist_sparce.py --num_epochs 50 --batch_size 16 --dim_res 2000 --perc_n 50.0 --prop_rec 0.1 --seed 42 &
python mnist_sparce.py --num_epochs 50 --batch_size 16 --dim_res 2000 --perc_n 75.0 --prop_rec 0.1 --seed 42 &
python mnist_sparce.py --num_epochs 50 --batch_size 16 --dim_res 2000 --perc_n 90.0 --prop_rec 0.1 --seed 42

wait

python mnist_sparce.py --num_epochs 50 --batch_size 16 --dim_res 2000 --perc_n 50.0 --prop_rec 0.2 --seed 42 &
python mnist_sparce.py --num_epochs 50 --batch_size 16 --dim_res 2000 --perc_n 75.0 --prop_rec 0.2 --seed 42 &
python mnist_sparce.py --num_epochs 50 --batch_size 16 --dim_res 2000 --perc_n 90.0 --prop_rec 0.2 --seed 42

wait

python mnist_sparce.py --num_epochs 50 --batch_size 16 --dim_res 2000 --perc_n 50.0 --prop_rec 0.5 --seed 42 &
python mnist_sparce.py --num_epochs 50 --batch_size 16 --dim_res 2000 --perc_n 75.0 --prop_rec 0.5 --seed 42 &
python mnist_sparce.py --num_epochs 50 --batch_size 16 --dim_res 2000 --perc_n 90.0 --prop_rec 0.5 --seed 42
