cd LightGBM-3.3.5
mkdir build
cd build
cmake ..
make
cd ../python-package/
python setup.py install
python -c """import lightgbm as lgb
if lgb.__version__ == \"3.3.5\":
    print(\" lightgbm v3.3.5 is installed\")
"""

cd ../..
cd xgboost
mkdir build
cd build
cmake ..
make
cd ../python-package/
python setup.py install
python -c """import xgboost as xgb
if xgb.__version__ == \"2.0.0-dev\":
    print(\" xgboost 2.0.0-dev is installed\")
"""
# cd ../../test
# bash test_all.sh
# cd ../..
# cd LightGBM-3.3.5/np_fair_test
# python example_baseline.py
# python example_fair.py
# python example_np.py

# cd ../..
# cd xgboost/np_fair_test
# python example_baseline.py
# python example_fair.py
# python example_np.py