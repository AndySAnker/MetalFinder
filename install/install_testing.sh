conda config --add channels diffpy
conda install --file "requirements_conda.txt"
pip3 install -r "Testing_requirements.txt" || pip install -r "Testing_requirements.txt"