@echo off
echo Starting Digital Twin Load Balanced Backend...

start "Worker 1" cmd /k "cd /d C:\Users\rahul\PycharmProjects\digital_twin_ml && C:\Users\rahul\PycharmProjects\digital_twin_ml\.venv\Scripts\uvicorn.exe main:app --host 127.0.0.1 --port 8001"
timeout /t 2
start "Worker 2" cmd /k "cd /d C:\Users\rahul\PycharmProjects\digital_twin_ml && C:\Users\rahul\PycharmProjects\digital_twin_ml\.venv\Scripts\uvicorn.exe main:app --host 127.0.0.1 --port 8002"
timeout /t 2
start "Worker 3" cmd /k "cd /d C:\Users\rahul\PycharmProjects\digital_twin_ml && C:\Users\rahul\PycharmProjects\digital_twin_ml\.venv\Scripts\uvicorn.exe main:app --host 127.0.0.1 --port 8003"

echo All 3 workers started
pause