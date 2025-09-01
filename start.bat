@echo off
echo Starting Teams Profil KI...

echo.
echo Starting PostgreSQL database...
docker-compose up -d

echo.
echo Waiting for database to be ready...
timeout /t 10 /nobreak > nul

echo.
echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Running application...
python main.py

echo.
echo Done!
