#!/bin/bash

source .venv/bin/activate

### PD POP
# python plot_from_agg.py --game pd --strategy pop --game-param 'b=1.8' --plot-type timeseries --theta 4.5 &
# python plot_from_agg.py --game pd --strategy pop --game-param 'b=1.8' --plot-type efficiency &

# Different b values
python plot_from_agg.py --agg-dir data_agg_cpp --fig-prefix cpp --game pd --strategy pop --game-param 'b=1.2' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --agg-dir data_agg_cpp --fig-prefix cpp --game pd --strategy pop --game-param 'b=1.2' --plot-type efficiency &
python plot_from_agg.py --agg-dir data_agg_cpp --fig-prefix cpp --game pd --strategy pop --game-param 'b=1.8' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --agg-dir data_agg_cpp --fig-prefix cpp --game pd --strategy pop --game-param 'b=1.8' --plot-type efficiency &
python plot_from_agg.py --agg-dir data_agg_cpp --fig-prefix cpp --game pd --strategy pop --game-param 'b=2.0' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --agg-dir data_agg_cpp --fig-prefix cpp --game pd --strategy pop --game-param 'b=2.0' --plot-type efficiency &

python plot_from_agg.py --agg-dir data_agg_cpp --fig-prefix cpp --game pd --strategy neb --game-param 'b=1.2' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --agg-dir data_agg_cpp --fig-prefix cpp --game pd --strategy neb --game-param 'b=1.2' --plot-type efficiency &
python plot_from_agg.py --agg-dir data_agg_cpp --fig-prefix cpp --game pd --strategy neb --game-param 'b=1.8' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --agg-dir data_agg_cpp --fig-prefix cpp --game pd --strategy neb --game-param 'b=1.8' --plot-type efficiency &
python plot_from_agg.py --agg-dir data_agg_cpp --fig-prefix cpp --game pd --strategy neb --game-param 'b=2.0' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --agg-dir data_agg_cpp --fig-prefix cpp --game pd --strategy neb --game-param 'b=2.0' --plot-type efficiency &

### PGG -- PYTHON
python plot_from_agg.py --agg-dir data_agg --fig-prefix py --game pgg --strategy pop --game-param 'r=1.5' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --agg-dir data_agg --fig-prefix py --game pgg --strategy pop --game-param 'r=1.5' --plot-type efficiency &
python plot_from_agg.py --agg-dir data_agg --fig-prefix py --game pgg --strategy pop --game-param 'r=3.0' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --agg-dir data_agg --fig-prefix py --game pgg --strategy pop --game-param 'r=3.0' --plot-type efficiency &
python plot_from_agg.py --agg-dir data_agg --fig-prefix py --game pgg --strategy pop --game-param 'r=4.5' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --agg-dir data_agg --fig-prefix py --game pgg --strategy pop --game-param 'r=4.5' --plot-type efficiency &

python plot_from_agg.py --agg-dir data_agg --fig-prefix py --game pgg --strategy neb --game-param 'r=1.5' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --agg-dir data_agg --fig-prefix py --game pgg --strategy neb --game-param 'r=1.5' --plot-type efficiency &
python plot_from_agg.py --agg-dir data_agg --fig-prefix py --game pgg --strategy neb --game-param 'r=3.0' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --agg-dir data_agg --fig-prefix py --game pgg --strategy neb --game-param 'r=3.0' --plot-type efficiency &
python plot_from_agg.py --agg-dir data_agg --fig-prefix py --game pgg --strategy neb --game-param 'r=4.5' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --agg-dir data_agg --fig-prefix py --game pgg --strategy neb --game-param 'r=4.5' --plot-type efficiency &

### PGG -- GO
python plot_from_agg.py --agg-dir data_agg_go --fig-prefix go --game pgg --strategy pop --game-param 'r=1.5' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --agg-dir data_agg_go --fig-prefix go --game pgg --strategy pop --game-param 'r=1.5' --plot-type efficiency &
python plot_from_agg.py --agg-dir data_agg_go --fig-prefix go --game pgg --strategy pop --game-param 'r=3.0' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --agg-dir data_agg_go --fig-prefix go --game pgg --strategy pop --game-param 'r=3.0' --plot-type efficiency &
python plot_from_agg.py --agg-dir data_agg_go --fig-prefix go --game pgg --strategy pop --game-param 'r=4.5' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --agg-dir data_agg_go --fig-prefix go --game pgg --strategy pop --game-param 'r=4.5' --plot-type efficiency &

python plot_from_agg.py --agg-dir data_agg_go --fig-prefix go --game pgg --strategy neb --game-param 'r=1.5' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --agg-dir data_agg_go --fig-prefix go --game pgg --strategy neb --game-param 'r=1.5' --plot-type efficiency &
python plot_from_agg.py --agg-dir data_agg_go --fig-prefix go --game pgg --strategy neb --game-param 'r=3.0' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --agg-dir data_agg_go --fig-prefix go --game pgg --strategy neb --game-param 'r=3.0' --plot-type efficiency &
python plot_from_agg.py --agg-dir data_agg_go --fig-prefix go --game pgg --strategy neb --game-param 'r=4.5' --plot-type timeseries --theta 4.5 &
python plot_from_agg.py --agg-dir data_agg_go --fig-prefix go --game pgg --strategy neb --game-param 'r=4.5' --plot-type efficiency &

### PGG -- DIFF
python plot_from_agg.py --agg-dir data_agg_go --fig-prefix go --game pgg --strategy pop --game-param 'r=1.5' --plot-type diff &
python plot_from_agg.py --agg-dir data_agg_go --fig-prefix go --game pgg --strategy pop --game-param 'r=3.0' --plot-type diff &
python plot_from_agg.py --agg-dir data_agg_go --fig-prefix go --game pgg --strategy pop --game-param 'r=4.5' --plot-type diff &

python plot_from_agg.py --agg-dir data_agg_go --fig-prefix go --game pgg --strategy neb --game-param 'r=1.5' --plot-type diff &
python plot_from_agg.py --agg-dir data_agg_go --fig-prefix go --game pgg --strategy neb --game-param 'r=3.0' --plot-type diff &
python plot_from_agg.py --agg-dir data_agg_go --fig-prefix go --game pgg --strategy neb --game-param 'r=4.5' --plot-type diff &

wait
