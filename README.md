# gvfs

## installing requirements
you will need to install the mazelab module from
https://github.com/zuoxingdong/mazelab

Do not forget to 
` python setup.py install`

## create ssh key and upload it on the git repo
open a terminal, and write command and press enter 3 times  
`ssh-keygen`

this will create a ssh key present in `~/.ssh/`. Go to that file  
`cd ~/.ssh/`  

Now, you have to add that key to your github account. This is to give your  
local computer the necessary permissions to interact with github. go to  
`https://github.com/settings/keys`  
press on `New ssh key` 
and copy paste what you have in the `~/.ssh/id_rsa.pub` file.  

Now you should have the permissions.  

## to set up the git
create a folder gvfs  

initialiaze git  
`git init`  

link to git  
`git remote add origin git@github.com:pablotanoretamales/gvfs.git`  

pull repo to have everything on your computer  
`git pull origin main`  
 
And you are set to go.  

## useful git commands:
`git status`: differences between your local code and the github one  
`git add`: add a file to be upload  
`git commit -m "describe what you do in the commit"`: commit your file adding  
`git push origin main`: push your files to github  


