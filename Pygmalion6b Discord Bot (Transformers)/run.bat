@echo off

rem Check if python is installed
where python
if %errorlevel% neq 0 (
    echo Python is not installed on this computer. Please install Python and run this script again.
    pause
    exit /b
)

rem Check if virtualenv is installed
pip show virtualenv
if %errorlevel% neq 0 (
    echo virtualenv is not installed on this computer. Installing virtualenv...
    pip install virtualenv
)

rem Create the virtual environment
virtualenv venv



rem Activate the virtual environment
call venv\Scripts\activate

rem Install the required packages
pip install -r requirements.txt
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

rem clear console
cls
rem Run the code
python discordbot.py

rem Deactivate the virtual environment
call venv\Scripts\deactivate

pause
