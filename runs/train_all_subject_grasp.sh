python main.py --dataset grasp --all_subject --model=bci2021  --labels=0,1 --start_time=-0.5 --end_time=4 --window_size=400 --step=50 -lr=1e-04 -sch=exp --gamma=0.993 --epochs=400 --batch_size=50 --save_dir=train --device=0
# salloc -N 1 -n 1 -p critical -t 240:00:00 --gres=gpu:1