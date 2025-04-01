:: Windows installer script for PyNetDesign
:: This script creates a Conda environment based on the "environment.yml" or "environment-dev.yml" file and installs the package
:: Run: install.bat from cmd
:: D. Anikiev, 2025-04-01

@echo off

set ENV_NAME=pnd
set PACKAGE_NAME=pynetdesign

:: Ask the user which environment to install
echo Choose which environment to install:
echo [1] User environment: %ENV_NAME% (uses environment.yml)
echo [2] Developer environment: %ENV_NAME%-dev (uses environment-dev.yml)
set /p ENV_CHOICE="Enter 1 for user or 2 for developer: "

if "%ENV_CHOICE%"=="1" (
    set ENV_YAML=environment.yml
    echo You chose to install the user environment.
) else if "%ENV_CHOICE%"=="2" (
    set ENV_NAME=%ENV_NAME%-dev
    set ENV_YAML=environment-dev.yml
    echo You chose to install the developer environment.
) else (
    echo Invalid choice. Exiting.
    pause
    exit /b 1
)

:: Check for Conda Installation
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Conda is not installed or not found in the system PATH.
    echo Please install Conda and make sure it's added to the system PATH.
    pause
    exit /b 2
)

:: Check for environment file
if not exist %ENV_YAML% (
    echo %ENV_YAML% not found in the current directory
    echo Please check and try again.
    pause
    exit /b 3
)

echo Creating Conda environment %ENV_NAME% from %ENV_YAML%...
call conda env create -f %ENV_YAML%
:: Check if environment creation was successful
if %ERRORLEVEL% equ 0 (
    echo Conda environment %ENV_NAME% created successfully.
) else (
    echo Failed to create Conda environment %ENV_NAME%. Please check the error messages above.
    pause
    exit /b 4
)

:: List conda environments
call conda env list

:: Activate environment
echo Activating Conda environment %ENV_NAME%...
call conda activate %ENV_NAME%
:: Check if environment activation was successful
if %ERRORLEVEL% equ 0 (    
    echo Conda environment %ENV_NAME% activated successfully.
) else (
    echo Failed to activate Conda environment %ENV_NAME%. Please check the error messages above.
    pause
    exit /b 5
)

echo Installing %PACKAGE_NAME%...
call pip install -e .
if %ERRORLEVEL% equ 0 (    
    echo Successfully installed %PACKAGE_NAME%.
) else (
    echo Failed to install %PACKAGE_NAME%. Please check the error messages above.
    pause
    exit /b 6
)

echo Python version:
call python --version

echo Python path:
:: Pick only first output
for /f "tokens=* usebackq" %%f in (`where python`) do (set "pythonpath=%%f" & goto :next)
:next

echo %pythonpath%

echo Done!

pause
