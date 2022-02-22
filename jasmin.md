# Useful links and commands for accessing JASMIN
<br/>

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

$ ssh-add -l [TO CHECK IT’S BEEN LOADED CORRECTLY]
<br/>
Logging into the JASMIN Servers:

$ ssh -A <user_id>@<login_server>

e.g. ssh -A usr12@login1.jasmin.ac.uk

