# Chat Playground - Gemini Launcher
Write-Host "Starting Chat Playground with Poetry..." -ForegroundColor Green
Write-Host ""

# Check if Poetry is installed
try {
    $poetryVersion = poetry --version
    Write-Host "Found Poetry: $poetryVersion" -ForegroundColor Cyan
} catch {
    Write-Host "Poetry not found! Please install Poetry first." -ForegroundColor Red
    Write-Host "Installation guide: https://python-poetry.org/docs/#installation"
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if dependencies are installed
if (-not (Test-Path "poetry.lock")) {
    Write-Host "Dependencies not installed. Running 'poetry install'..." -ForegroundColor Yellow
    poetry install
}

# Run the Streamlit app
Write-Host "Launching Streamlit app..." -ForegroundColor Green
poetry run streamlit run chat_main.py 