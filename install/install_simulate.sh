conda config --add channels diffpy
conda install --file "requirements_conda.txt"
pip3 install -r "Simulate_Data_requirements.txt" || pip install -r "Simulate_Data_requirements.txt"