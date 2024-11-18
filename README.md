# README

To run the code locally using Anaconda:

1. Install Anaconda:
   - Download and install Anaconda from [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution).

2. Create a new Conda environment with Python 3.10:
   ```
   conda create -n trading_env_new python=3.10
   ```

3. Activate the new environment:
   ```
   conda activate trading_env_new
   ```

4. Install necessary libraries using Conda:
   ```
   conda install numpy=1.26.4 pandas=2.2.2 matplotlib scikit_learn
   ```

5. Install additional libraries using pip:
   ```
   pip install tensorflow==2.10.1 yfinance
   pip install tensorflow-addons==0.18.0
   pip install gym stable_baselines3
   pip install shimmy==0.2.1
   pip install ta
   ```
