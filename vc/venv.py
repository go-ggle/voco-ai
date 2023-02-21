import subprocess
from subprocess import Popen, PIPE
from subprocess import check_output
from flask import Flask

def get_shell_script_output_using_communicate():
    session = Popen(['./stargan_venv.sh'], stdout=PIPE, stderr=PIPE)
    stdout, stderr = session.communicate()
    if stderr:
        raise Exception("Error "+str(stderr.decode('utf-8')))
    return stdout.decode('utf-8')

def get_shell_script_output_using_check_output():
    stdout = check_output(['./stargan_venv.sh']).decode('utf-8')
    return stdout

app = Flask(__name__)

@app.route('/venv',methods=['GET',])
def home():
    return '<pre>'+get_shell_script_output_using_check_output()+'</pre>'

app.run(debug=True)
