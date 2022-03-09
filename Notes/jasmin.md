# Useful links and commands for accessing JASMIN

## Links: <br/>

JASMIN Login:
https://accounts.jasmin.ac.uk

JASMIN BAS Area Request:
https://accounts.jasmin.ac.uk/services/group_workspaces/?query=bas_climate

JASMIN Notebook Services:
https://notebooks.jasmin.ac.uk/

How to log into JASMIN Servers:
https://help.jasmin.ac.uk/article/187-login

<br/>

## Adding SSH Key:

	$ eval $(ssh-agent -s)

	$ ssh-add ~/.ssh/id_rsa_jasmin

	$ ssh-add -l [to check it's been correctly loaded]

<br/>

## Logging into the JASMIN Servers:

	$ ssh -A <user_id>@<login_server>

	e.g. ssh -A usr12@login1.jasmin.ac.uk


	$ logout (to logout)

<br/>

Once logged in, JASMIN message will show usage of each VM over past hour. To log into specific VM use:

	$ ssh <user_id>@<VM name>
  
	e.g. ssh user12@sci3.jasmin.ac.uk

To prevent timeout of the SSH connection, set the ServerAliveInterval to send alive signals every 30 seconds:

	$ ssh -o ServerAliveInterval=30 <username>@<VM name>
	
	e.g. ssh -o ServerAliveInterval=30 user12@sci3.jasmin.ac.uk

This can also be done in the initial log in as:

	$ ssh -o ServerAliveInterval=30 -A <user_id>@<login_server>

	e.g. ssh -o ServerAliveInterval=30 -A usr12@login1.jasmin.ac.uk


Once logged in to specific VM (may take a few ports), can link with Jupyter + porgrams with:
  
	module load jaspy

<br/>

## Navigating to Shared Workspace:
	
	$ cd /
	
	$ cd gws/nopw/j04/bas_climate/projects/SeaIceExtent

<br/>

## Data Transfer:

Transferring to your home directory:

	$ rsync filename_for_transfer <user_id>@<transfer_server>:/home/users/user_id/
	
	e.g. rsync jonnycode.py usr12@xfer1.jasmin.ac.uk:/home/users/usr12/


Transferring from your home directory:
	
	[Need to be have ssh key for jasmin added, and be in local area on laptop]
	
	$ rsync <user_id>@<transfer_server>:/home/users/user_id/filename .
	
	
	e.g. eval $(ssh-agent -s)
	
	ssh-add ~/.ssh/id_rsa_jasmin

	rsync jdr53@xfer1.jasmin.ac.uk:/home/users/jdr53/transfer_test.txt <local_filepath>

Transfering within JASMIN
	
	$ cp file.file destination
	
	e.g. cp functions.py /home/users/user12

<br/>	

## Creating virtual environment on JASMIN:

Navigate to venv area
	
	# Load jaspy to use venv command
	$ module load jaspy
	
	# Create a virtual environment, including all the jaspy libraries
	$ python -m venv --system-site-packages ./<env_name>
	
	# Activate the env
	$ source ./env_name/bin/activate

	# Install packages as needed
	$ pip install <package_name>
	
For our project we have installed: rasterio, wandb, torch, torchIO, segmentation_models_pytorch

<br/>

Navigating to personal space on JASMIN:

	# Navigate to personal directory
	$ cd /
	$ home/users/<username>

<br/>

Other useful commands:

	# Shut down/logout
	$ exit
	
	# To change read/write:
	$ chmod g+w <filepath>
	e.g. chmod g+w /gws/nopw/j04/bas_climate/projects/SeaIceExtent
	
	# To create a TMP file locally in JASMIN (in response to wandb ERROR 13)
	$ export TMPDIR="$(mktemp -d -t ci-XXXXXXX --tmpdir=/home/users/<user>/tempfiles)"

<br/>

## Group Workspace Location:
	
	# /gws/nopw/j04/bas_climate/projects/SeaIceExtent

<br/>

## Slurm:

<br/>

Slurm is a job scheduling manager. On JASMIN it seems like the easiest way to submit a job is via a batch script.

The batch script should follow the same template, although additional flags/options can be added. The template is as follows:

<br/>

	#!/bin/bash
	#SBATCH --partition=short-serial
	#SBATCH -o %j.out
	#SBATCH -e %j.err
	#SBATCH --time=01:00:00
	#SBATCH --ntasks=50

	# executable
	python3 tune_hyperparameters_jasmin.py

<br/>

The .out and .err files record the output from python (e.g. prints) and the output from the console, respectively.

--time refers to the number of hours allocated for the job; the format is: hh:mm:ss.

--ntasks refers to the number of cores required for the job. I think this is a balance between using too much of the compute power and using too few and having a slower runtime.


The batch script should have the extension .sh, e.g. test_job.sh, and can be submitted as:

	$ sbatch test_job.sh


You can view your submitted jobs using the following:

	$ squeue -u <username> ; e.g. squeue -u jdr53


This will display a list of the jobs you have submitted that are currently active, along with a jobID for each.


The following command can be used to see more details for a particular job:


	$ scontrol show job <jobID>

## SLURM LOTUS GPU Cluster
<br/>
After being added to the LOTUS gpu users group it is possible to run scripts using the GPUs, provided a few minor changes are made to the batch scripts.

Essential changes are changing the partition and account to lotus_gpu, and adding a gres=gpu:1 line. For performance and to try and reduce out of memory errors, the following should also be added:
 --ntasks=32 ; --ntasks-per-node=32 ; --mem=32000

The lotus_gpu partition has a runtime limit of 24 hours (much reduced from the purported 168hrs runtime on their webpages).

Here is an example batch script for the GPU cluser.

#!/bin/bash
#SBATCH --partition=lotus_gpu
#SBATCH --account=lotus_gpu
#SBATCH --gres=gpu:1
#SBATCH -o outputname.out
#SBATCH -e outputname.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --mem=32000
# executable
python3 python_script.py
