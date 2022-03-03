# Useful links and commands for accessing JASMIN

Links: <br/>

JASMIN Login:
https://accounts.jasmin.ac.uk

JASMIN BAS Area Request:
https://accounts.jasmin.ac.uk/services/group_workspaces/?query=bas_climate

JASMIN Notebook Services:
https://notebooks.jasmin.ac.uk/

How to log into JASMIN Servers:
https://help.jasmin.ac.uk/article/187-login

<br/>

Adding SSH Key:

	$ eval $(ssh-agent -s)

	$ ssh-add ~/.ssh/id_rsa_jasmin

	$ ssh-add -l [to check it's been correctly loaded]

<br/>

Logging into the JASMIN Servers:

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

Navigating to Shared Workspace:
	
	$ cd /
	
	$ cd gws/nopw/j04/bas_climate/projects/SeaIceExtent

<br/>

Data Transfer:

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

Creating virtual environment on JASMIN:

Navigate to venv area
	
	# Load jaspy to use venv command
	$ module load jaspy
	
	# Create a virtual environment, including all the jaspy libraries
	$ python -m venv --system-site-packages ./<env_name>
	
	# Activate the env
	$ source ./env_name/bin/activate

	# Install packages as needed
	$ pip install <package_name>

<br/>

Navigating to personal space on JASMIN:

	# Navigate to personal directory
	$ cd /
	$ home/users/<username>

<br/>

Other useful commands:

	# Shut down/logout
	$ exit


<br/>

Group Workspace Location:
	
	# /gws/nopw/j04/bas_climate/projects/SeaIceExtent
