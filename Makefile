HOST ?= 0.0.0.0
PORT ?= 5173
PID_FILE ?= .vite.pid
LOG_FILE ?= .vite.log
VITE_BIN := ./node_modules/.bin/vite

.DEFAULT_GOAL := help

.PHONY: help install run stop restart status logs build

help:
	@echo "Available targets:"
	@echo "  make install   Install dependencies"
	@echo "  make run       Start the app in the background"
	@echo "  make stop      Stop the app"
	@echo "  make restart   Restart the app"
	@echo "  make status    Show app status"
	@echo "  make logs      Show dev server logs"
	@echo "  make build     Build the app"

install:
	npm install

run:
	@if [ ! -x "$(VITE_BIN)" ]; then \
		echo "Dependencies are missing. Run: make install"; \
		exit 1; \
	fi
	@if [ -f "$(PID_FILE)" ] && kill -0 "$$(cat "$(PID_FILE)")" 2>/dev/null; then \
		echo "App is already running at http://localhost:$(PORT) (PID $$(cat "$(PID_FILE)"))"; \
	else \
		app_dir="$$(basename "$$(pwd)")"; \
		pids="$$(pgrep -f "$$app_dir.*vite" || true)"; \
		if [ -n "$$pids" ]; then \
			set -- $$pids; \
			echo "$$1" > "$(PID_FILE)"; \
			echo "App is already running at http://localhost:$(PORT) (PID $$1)"; \
		else \
			rm -f "$(PID_FILE)"; \
			echo "Starting app at http://localhost:$(PORT)"; \
			nohup "$(VITE_BIN)" --host "$(HOST)" --port "$(PORT)" > "$(LOG_FILE)" 2>&1 & \
			echo $$! > "$(PID_FILE)"; \
			sleep 1; \
			if kill -0 "$$(cat "$(PID_FILE)")" 2>/dev/null; then \
				echo "Started app (PID $$(cat "$(PID_FILE)")). Logs: $(LOG_FILE)"; \
			else \
				echo "App failed to start. Logs:"; \
				cat "$(LOG_FILE)"; \
				rm -f "$(PID_FILE)"; \
				exit 1; \
			fi; \
		fi; \
	fi

stop:
	@if [ -f "$(PID_FILE)" ] && kill -0 "$$(cat "$(PID_FILE)")" 2>/dev/null; then \
		echo "Stopping app (PID $$(cat "$(PID_FILE)"))"; \
		kill "$$(cat "$(PID_FILE)")"; \
		rm -f "$(PID_FILE)"; \
	else \
		rm -f "$(PID_FILE)"; \
		app_dir="$$(basename "$$(pwd)")"; \
		pids="$$(pgrep -f "$$app_dir.*vite" || true)"; \
		if [ -n "$$pids" ]; then \
			echo "Stopping app processes: $$pids"; \
			kill $$pids; \
		else \
			echo "App is not running"; \
		fi; \
	fi

restart: stop run

status:
	@if [ -f "$(PID_FILE)" ] && kill -0 "$$(cat "$(PID_FILE)")" 2>/dev/null; then \
		echo "App is running at http://localhost:$(PORT) (PID $$(cat "$(PID_FILE)"))"; \
	else \
		app_dir="$$(basename "$$(pwd)")"; \
		pids="$$(pgrep -f "$$app_dir.*vite" || true)"; \
		if [ -n "$$pids" ]; then \
			echo "App is running at http://localhost:$(PORT) (PID(s) $$pids)"; \
		else \
			echo "App is not running"; \
		fi; \
	fi

logs:
	@if [ -f "$(LOG_FILE)" ]; then \
		tail -f "$(LOG_FILE)"; \
	else \
		echo "No log file found: $(LOG_FILE)"; \
	fi

build:
	npm run build
