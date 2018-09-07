# Learning Adversarially Fair and Transferable Representations
David Madras, Elliot Creager, Toni Pitassi, Richard Zemel
https://arxiv.org/abs/1802.06309

Code represents equal contribution with [David Madras](https://github.com/dmadras/).
Thanks to [Jake Snell](https://github.com/jakesnell) and [James Lucas](https://github.com/AtheMathmo) for contributing the experiment sweep code.

## setting up a project-specific virtual env
```
mkdir ~/venv 
python3 -m venv ~/venv/laftr
```
where `python3` points to python 3.6.X. Then
```
source ~/venv/laftr/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
or 
```
pip install -r requirements-gpu.txt
```
for GPU support

## running a single fair classification experiment
```
source simple_example.sh
```
The bash script first trains LAFTR and then evaluates by training a naive classifier on the LAFTR representations (encoder outputs).
See the paper for further details.

## running a sweep of fair classification with various hyperparameter values
```
python src/generate_sweep.py sweeps/small_sweep_adult/sweep.json
source sweeps/small_sweep_adult/command.sh
```
The above script is a small sweep which only trains for a few epochs. 
It is basically just for making sure everything runs smoothly. 
For a bigger sweep call `src/generate_sweep` with `sweeps/full_sweep_adult/sweep.json`, or design your own sweep config.

## data
The (post-processed) adult dataset is provided in `data/adult/adult.npz`

