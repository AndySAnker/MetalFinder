conda config --add channels diffpy
conda install --file "../requirement_files/requirements_conda.txt"
pip3 install -r "../requirement_files/Testing_requirements.txt" || pip install -r "../requirement_files/Testing_requirements.txt"