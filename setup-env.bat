CALL py -3.9 -m venv myenv
CALL myenv\Scripts\activate
CALL pip install -r requirements.txt
echo "Environment setup complete, please run 'start-fluffy' to start the program."
PAUSE