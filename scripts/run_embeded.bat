set PWD=%~dp0
.\python_embeded\python.exe webui\app.py --chdir %PWD%\webui --userdata %PWD%\userdata 
pause