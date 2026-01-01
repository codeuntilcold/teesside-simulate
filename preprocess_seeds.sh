#!/bin/bash

source .venv/bin/activate

python preprocess_seeds.py --data-dir data/pop_theta --output-dir data_agg &

### theta 4-5

# python preprocess_seeds.py --data-dir 'data/pop_theta_pgg_r=1.5' --output-dir data_agg &
# python preprocess_seeds.py --data-dir 'data/pop_theta_pgg_r=3.0' --output-dir data_agg &
# python preprocess_seeds.py --data-dir 'data/pop_theta_pgg_r=4.5' --output-dir data_agg &
# python preprocess_seeds.py --data-dir 'data/neb_theta_pgg_r=1.5' --output-dir data_agg &
# python preprocess_seeds.py --data-dir 'data/neb_theta_pgg_r=3.0' --output-dir data_agg &
# python preprocess_seeds.py --data-dir 'data/neb_theta_pgg_r=4.5' --output-dir data_agg &

# theta 0-10; python missing neb

python preprocess_seeds.py --data-dir 'data_py_theta010/pop_theta_pgg_r=1.5' --output-dir data_agg &
python preprocess_seeds.py --data-dir 'data_py_theta010/pop_theta_pgg_r=3.0' --output-dir data_agg &
python preprocess_seeds.py --data-dir 'data_py_theta010/pop_theta_pgg_r=4.5' --output-dir data_agg &
python preprocess_seeds.py --data-dir 'data_py_theta010/neb_theta_pgg_r=1.5' --output-dir data_agg &
python preprocess_seeds.py --data-dir 'data_py_theta010/neb_theta_pgg_r=3.0' --output-dir data_agg &
python preprocess_seeds.py --data-dir 'data_py_theta010/neb_theta_pgg_r=4.5' --output-dir data_agg &

# python preprocess_seeds.py --data-dir data_go_theta010/pop_theta_pgg_r=1.5 --output-dir data_agg &
# python preprocess_seeds.py --data-dir data_go_theta010/pop_theta_pgg_r=3   --output-dir data_agg &
# python preprocess_seeds.py --data-dir data_go_theta010/pop_theta_pgg_r=4.5 --output-dir data_agg &
# python preprocess_seeds.py --data-dir data_go_theta010/neb_theta_pgg_r=1.5 --output-dir data_agg &
# python preprocess_seeds.py --data-dir data_go_theta010/neb_theta_pgg_r=3   --output-dir data_agg &
# python preprocess_seeds.py --data-dir data_go_theta010/neb_theta_pgg_r=4.5 --output-dir data_agg &

wait
