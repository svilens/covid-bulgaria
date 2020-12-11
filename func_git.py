import subprocess as cmd

def git_push_automation():
    try:
        #cp = cmd.run("file path", check=True, shell=True)
        #logger.info("cp", cp)
        cmd.run('git add -A', check=True, shell=True)
        cmd.run('git commit -m "daily update"', check=True, shell=True)
        cmd.run("git push heroku master", check=True, shell=True)
        logger.info("Pushed to Heroku")
        return True
    except:
        logger.info("Git automation failed")
        return False
    
    
from git import Repo

def git_push_heroku(path, commit_message):
    try:
        repo = Repo(path)
        repo.git.add(update=True)
        repo.index.commit(commit_message)
        origin = repo.remote(name='heroku')
        origin.push()
        result_message = "Pushed to Heroku"
    except:
        result_message = 'Some error occured while pushing'
    return result_message
