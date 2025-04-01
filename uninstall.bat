:: Windows uninstaller script for PyNetDesign
:: This script removes associated Conda environments matching "pnd*"
:: Run: uninstall.bat from cmd
:: D. Anikiev, 2025-04-01

@echo off

set ENV_PATTERN=pnd
set ENV_FOUND=0

:: Check for Conda Installation
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Conda is not installed or not found in the system PATH.
    echo Please install Conda and make sure it's added to the system PATH.
    pause
    exit /b 1
)

:: Find environments matching the pattern "pnd*"
for /f "tokens=*" %%A in ('conda info --envs ^| findstr /r /c:"%ENV_PATTERN%.*"') do (
    for /f "tokens=1" %%B in ("%%A") do (
        set "ENV_NAME=%%B"
        set "ENV_FOUND=1"
        echo Found Conda environment: "%ENV_NAME%"

        :: Ask user if they want to remove each environment
        set /p CONFIRM="Do you want to remove environment %ENV_NAME%? (y/n): "
        if /i "%CONFIRM%" == "y" (
            :: Deactivate any active environment before removal
            echo Deactivating any active Conda environment...
            call conda deactivate

            :: Remove the environment
            echo Removing Conda environment %ENV_NAME%...
            call conda remove -n %ENV_NAME% --all
            if %ERRORLEVEL% equ 0 (
                echo Conda environment %ENV_NAME% removed successfully.
            ) else (
                echo Failed to remove Conda environment %ENV_NAME%. Please check the error messages above.
                pause
                exit /b 3
            )
        ) else (
            echo Skipping Conda environment %ENV_NAME%.
        )
    )
)

:: Check if no matching environments were found
if %ENV_FOUND% equ 0 (
    echo No Conda environments matching "%ENV_PATTERN%*" were found.
) else (
    echo Done!
)

pause
