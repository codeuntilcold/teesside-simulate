#!/bin/bash

source .venv/bin/activate

# PD POP
python plot_from_agg.py --game pd --strategy pop --game-param 'b=1.8' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --game pd --strategy pop --game-param 'b=1.8' --plot-type efficiency &

# PGG POP
python plot_from_agg.py --game pgg --strategy pop --game-param 'r=1.5' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --game pgg --strategy pop --game-param 'r=1.5' --plot-type efficiency &
python plot_from_agg.py --game pgg --strategy pop --game-param 'r=3.0' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --game pgg --strategy pop --game-param 'r=3.0' --plot-type efficiency &
python plot_from_agg.py --game pgg --strategy pop --game-param 'r=4.5' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --game pgg --strategy pop --game-param 'r=4.5' --plot-type efficiency &

# PGG NEB
python plot_from_agg.py --game pgg --strategy neb --game-param 'r=1.5' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --game pgg --strategy neb --game-param 'r=1.5' --plot-type efficiency &
python plot_from_agg.py --game pgg --strategy neb --game-param 'r=3.0' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --game pgg --strategy neb --game-param 'r=3.0' --plot-type efficiency &
python plot_from_agg.py --game pgg --strategy neb --game-param 'r=4.5' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --game pgg --strategy neb --game-param 'r=4.5' --plot-type efficiency &

wait
