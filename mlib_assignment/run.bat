@echo off
echo =====================================================
echo 🎵 Starting Lyrics Genre Prediction System...
echo =====================================================

REM Step 1: Create venv if not already created
IF NOT EXIST venv (
    echo 🔧 Creating virtual environment...
    python -m venv venv
)

REM Step 2: Activate the virtual environment
echo ✅ Activating virtual environment...
call venv\Scripts\activate.bat

REM Step 3: Install required packages (manual install)
echo 📦 Installing Flask, PySpark, and FindSpark...
pip install flask pyspark findspark numpy

REM Step 4: Run the server
cd webapp
echo 🚀 Launching server...
python server.py

pause
