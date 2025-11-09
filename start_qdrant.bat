@echo off
REM Start Qdrant vector database using Docker

echo Starting Qdrant...
echo.

REM Check if Docker is running
docker version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running or not installed.
    echo.
    echo Please start Docker Desktop first, then run this script again.
    echo Download Docker Desktop from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Check if container already exists
docker ps -a --format "{{.Names}}" | findstr /C:"qdrant" >nul 2>&1
if %errorlevel% equ 0 (
    echo Qdrant container found. Starting it...
    docker start qdrant
    if errorlevel 1 (
        echo Failed to start existing container.
        pause
        exit /b 1
    )
) else (
    echo Creating new Qdrant container...
    docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant
    if errorlevel 1 (
        echo Failed to create container.
        pause
        exit /b 1
    )
)

echo.
echo Qdrant is running!
echo.
echo API: http://localhost:6333
echo Dashboard: http://localhost:6334/dashboard
echo.
echo To stop: docker stop qdrant
echo To remove: docker rm qdrant
echo.
pause
