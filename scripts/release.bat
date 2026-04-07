@echo off
REM Release helper script for Windows
REM Usage: scripts\release.bat [version]

setlocal enabledelayedexpansion

if "%1"=="" (
    echo Error: Version number required
    echo Usage: scripts\release.bat 0.1.0
    exit /b 1
)

set VERSION=%1

echo ==========================================
echo Release Helper - Version %VERSION%
echo ==========================================

REM Step 1: Check current branch
echo.
echo [1/7] Checking current branch...
for /f "delims=" %%i in ('git branch --show-current') do set CURRENT_BRANCH=%%i
if not "!CURRENT_BRANCH!"=="main" (
    if not "!CURRENT_BRANCH!"=="master" (
        echo Warning: You are on branch '!CURRENT_BRANCH!'
        set /p CONTINUE="Continue? (y/n): "
        if /i not "!CONTINUE!"=="y" exit /b 1
    )
)

REM Step 2: Run tests
echo.
echo [2/7] Running tests...
python verify_installation.py
pytest tests/ -v
if errorlevel 1 (
    echo Tests failed!
    exit /b 1
)

REM Step 3: Update version in setup.py
echo.
echo [3/7] Updating version in setup.py...
powershell -Command "(Get-Content setup.py) -replace 'version=\"[^\"]*\"', 'version=\"%VERSION%\"' | Set-Content setup.py"

REM Step 4: Update version in __init__.py
echo.
echo [4/7] Updating version in __init__.py...
powershell -Command "(Get-Content src\time_series_forecasting_zero\__init__.py) -replace '__version__ = \"[^\"]*\"', '__version__ = \"%VERSION%\"' | Set-Content src\time_series_forecasting_zero\__init__.py"

REM Step 5: Commit changes
echo.
echo [5/7] Committing version changes...
git add setup.py src\time_series_forecasting_zero\__init__.py
git commit -m "Bump version to %VERSION%"

REM Step 6: Create tag
echo.
echo [6/7] Creating git tag v%VERSION%...
git tag -a "v%VERSION%" -m "Release version %VERSION%"

REM Step 7: Push
echo.
echo [7/7] Pushing to remote...
echo This will trigger CI/CD pipeline to:
echo   - Run tests on multiple platforms
echo   - Build distribution packages
echo   - Publish to PyPI
echo   - Create GitHub Release
echo.
set /p CONTINUE="Continue? (y/n): "
if /i "%CONTINUE%"=="y" (
    git push origin HEAD
    git push origin "v%VERSION%"
    echo.
    echo ✅ Release process started!
    echo Check progress at: https://github.com/yourusername/Time-Series-Forecasting-Zero/actions
) else (
    echo Cancelled. Tag and commit are local only.
    echo To push manually:
    echo   git push origin HEAD
    echo   git push origin v%VERSION%
)

echo.
echo ==========================================
echo Release preparation complete!
echo ==========================================

endlocal
