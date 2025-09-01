# Makefile for Teams Profil KI

.PHONY: start db-start db-stop db-restart install run clean help

# Default target
help:
	@echo "Teams Profil KI - Available commands:"
	@echo "  make start      - Start database and run application"
	@echo "  make db-start   - Start only the database"
	@echo "  make db-stop    - Stop the database"
	@echo "  make db-restart - Restart the database"
	@echo "  make install    - Install Python dependencies"
	@echo "  make run        - Run the application (database must be running)"
	@echo "  make clean      - Stop all services and clean up"

# Start everything
start: db-start install run

# Database operations
db-start:
	@echo "Starting PostgreSQL database..."
	docker-compose up -d
	@echo "Waiting for database to be ready..."
	@timeout /t 10 /nobreak > nul
	@echo "Database is ready!"

db-stop:
	@echo "Stopping database..."
	docker-compose down

db-restart: db-stop db-start

# Python operations
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt

run:
	@echo "Running application..."
	python main.py

# Cleanup
clean:
	@echo "Cleaning up..."
	docker-compose down -v
	@echo "Cleanup complete!"
