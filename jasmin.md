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
  
	e.g. ssh mcl65@sci3.jasmin.ac.uk


Once logged in to specific VM (may take a few ports), can link with Jupyter + porgrams with:
  
	module load jaspy

<br/>

Data Transfer:

Transferring to your home directory:

	$ rsync filename_for_transfer <user_id>@<transfer_server>:/home/users/user_id/
	
	e.g rsync jonnycode.py usr12@xfer1.jasmin.ac.uk:/home/users/usr12/


Transferring from your home directory:
	
	[Need to be have ssh key for jasmin added, and be in local area on laptop]
	
	$ rsync <user_id>@<transfer_server>:/home/users/user_id/filename .
	
	
	e.g. eval $(ssh-agent -s)
	
	ssh-add ~/.ssh/id_rsa_jasmin

	rsync jdr53@xfer1.jasmin.ac.uk:/home/users/jdr53/transfer_test.txt <local_filepath>

<br/>	

Creating virtual environment on JASMIN:

Navigate to area 
	
	# Load jaspy to use venv command
	$ module load jaspy
	
	# Create a virtual environment, including all the jaspy libraries
	$ python -m venv --system-site-packages ./<env_name>
	
	# Activate the env
	$ source ./env_name/bin/activate

	# Install packages as needed
	$ pip install <package_name>
