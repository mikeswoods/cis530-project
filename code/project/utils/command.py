import subprocess

################################################################################
# Command running and output capture utility functions
################################################################################

def run_command(*args)
    p = subprocess.Popen(*args, stdout=subprocess.PIPE)
    (out, err) = p.communicate()
