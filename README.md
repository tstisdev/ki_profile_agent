# Teams Profil KI - Development Setup

## Prerequisites
- Docker and Docker Compose
- Python 3.8+
- pip

## Quick Start

### Option 1: Using PyCharm
1. Open the project in PyCharm
2. Select "Start Database & Run App" from the run configuration dropdown
3. Click the green Run button

### Option 2: Using Command Line
```bash
# Start database
make db-start

# Install dependencies and run app
make run

# Or do everything at once
make start
```

### Option 3: Manual Setup
```bash
# Start PostgreSQL database
docker-compose up -d

# Install Python dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Available Commands

- `make start` - Start database and run application
- `make db-start` - Start only the database
- `make db-stop` - Stop the database
- `make db-restart` - Restart the database
- `make install` - Install Python dependencies
- `make run` - Run the application (database must be running)
- `make clean` - Stop all services and clean up

## Database Access

- **Host:** localhost
- **Port:** 5432
- **Database:** teams_profil_ki
- **Username:** postgres
- **Password:** password123

## Project Structure

- `Data/` - Input PDF files (gitignored)
- `Anonymized/` - Output anonymized PDFs (gitignored)
- `docker-compose.yml` - PostgreSQL database setup
- `database.py` - Database connection and operations
- `main.py` - Main application logic
