# Chat Playground - Gemini & Cerebras

A unified chat interface for Cerebras and Gemini AI models with advanced system prompt management capabilities.

## Features

- **Multi-Model Support**: Use both Cerebras and Gemini AI models
- **System Prompt Management**: Save, edit, delete, and switch between multiple system prompts
- **Modern UI**: Clean Streamlit interface with dark/light mode support
- **Chat History**: Upload/download chat sessions as JSON
- **Advanced Settings**: Temperature control, token limits, message history management

## Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)
- API Keys:
  - CEREBRAS_API_KEY (optional, for Cerebras models)
  - GEMINI_API_KEY (optional, for Gemini models)

## 🧠 **Unified Thinking Support**

Both **Cerebras Qwen models** (via `<think>` tags) and **Gemini 2.5 models** (via native thinking API) display their reasoning process in a unified "🧠 Thought Process" expandable section.

**Thinking-Capable Models:**
- **Cerebras**: `qwen-3-32b`, `llama-4-scout-17b-16e-instruct` 
- **Gemini**: `gemini-2.5-pro-002`, `gemini-2.5-flash-002`, `gemini-2.5-flash-8b-001`

### **Enhanced Thinking Features (Phase 3):**
- **Smart Model Grouping**: Thinking models (🧠) are separated from standard models (⚡)
- **Performance Indicators**: Each model shows capabilities like "🚀 Fast inference" or "📊 Complex analysis"
- **Thinking Settings Panel**: Control auto-expansion, statistics display, and verbosity
- **Enhanced Statistics**: Word count, character count, and thinking analysis
- **Auto-Expand Option**: Automatically open thinking sections for new responses
- **Responsive Display**: Thinking text area height adapts to content length

### **Model Selection Improvements:**
- **Visual Indicators**: 🧠 for thinking models, ⚡ for standard models
- **Grouped Display**: Thinking models appear first, separated from standard models
- **Capability Badges**: Performance indicators for each model type
- **Smart Descriptions**: Context-aware model information

## 🚀 **Quick Start**

### Option 1: Use Launcher Scripts (Easiest)

**Windows Batch File:**
```batch
# Double-click run_app.bat or in Command Prompt:
run_app.bat
```

**PowerShell Script:**
```powershell
# Right-click and "Run with PowerShell" or:
.\run_app.ps1
```

### Option 2: Manual Poetry Commands

```powershell
# Make sure you're in the project directory
cd "C:\Dev\ChatPlayground - Gemini"

# Install dependencies (first time only)
poetry install

# Run the application
poetry run streamlit run chat_main.py
```

## Poetry Setup (Already Configured!)

This project is now fully configured with Poetry! The `pyproject.toml` file contains all dependency information.

### Key Poetry Commands for This Project

**Activate Virtual Environment:**
```powershell
poetry shell
# Then you can run: streamlit run chat_main.py
```

**Install Dependencies:**
```powershell
poetry install
```

**Add New Dependencies:**
```powershell
poetry add package-name
```

**Add Development Dependencies:**
```powershell
poetry add --group dev package-name
```

**Update Dependencies:**
```powershell
poetry update
```

**View Virtual Environment Info:**
```powershell
poetry env info
```

## Environment Variables

### Local Development
Create a `.env` file in the project root:

```env
CEREBRAS_API_KEY=your_cerebras_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### Streamlit Cloud Deployment
Add to Streamlit secrets in your app settings:
```toml
CEREBRAS_API_KEY = "your_key_here"
GEMINI_API_KEY = "your_key_here"
```

## Project Structure

```
ChatPlayground - Gemini/
├── pyproject.toml          # Poetry configuration & dependencies
├── poetry.lock             # Locked dependency versions (auto-generated)
├── chat_main.py            # Main Streamlit application
├── cerebras_client.py      # Cerebras API client
├── gemini_client.py        # Gemini API client
├── system_prompts.json     # System prompts storage
├── system_prompt.txt       # Legacy prompt file
├── run_app.bat            # Windows batch launcher
├── run_app.ps1            # PowerShell launcher
├── .streamlit/            # Streamlit configuration
├── .env                   # Environment variables (create this)
├── .gitignore             # Git ignore rules
└── README.md             # This file
```

## Development Tools (Included)

### Code Formatting with Black
```powershell
poetry run black .
```

### Linting with Flake8
```powershell
poetry run flake8 .
```

### Testing with Pytest
```powershell
poetry run pytest
```

## Usage

1. **Select a Model**: Choose from available Cerebras or Gemini models
2. **Manage System Prompts**: 
   - Select from existing prompts
   - Edit current prompts
   - Add new prompts
   - Delete unwanted prompts
3. **Configure Settings**: Access advanced settings for temperature, token limits, etc.
4. **Chat**: Start conversing with the AI using your selected model and prompt

## System Prompt Management

The app supports multiple named system prompts stored in `system_prompts.json`:

- **Select**: Choose from a dropdown of saved prompts
- **Edit**: Modify the selected prompt's content
- **Add**: Create new prompts with custom names
- **Delete**: Remove prompts (minimum one must remain)

## Troubleshooting

### Poetry Issues
```powershell
# Clear poetry cache
poetry cache clear --all .

# Rebuild environment
poetry env remove python
poetry install
```

### Missing Dependencies
```powershell
poetry install --no-cache
```

### Streamlit Issues
```powershell
# Clear Streamlit cache
poetry run streamlit cache clear
```

### Permission Issues (PowerShell Script)
```powershell
# If you get execution policy errors:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Dependencies

**Main Dependencies:**
- `streamlit`

## 🌐 **Streamlit Community Edition Deployment**

This app is configured for easy deployment on Streamlit Community Edition.

### **Deployment Files:**
- **`requirements.txt`**: Contains all necessary dependencies for cloud deployment
- **`.streamlit/config.toml`**: Forces Poetry usage and sets app theme
- **`.streamlit/secrets.toml`**: Add your API keys here for deployment

### **Setting Up Secrets:**
In your Streamlit Community Edition app settings, add:
```toml
CEREBRAS_API_KEY = "your_cerebras_api_key_here"
GEMINI_API_KEY = "your_gemini_api_key_here"
```

### **Deployment Notes:**
- ✅ Both Poetry (`pyproject.toml`) and pip (`requirements.txt`) are supported
- ✅ Thinking functionality works in the cloud with `google-genai>=1.18.0`
- ✅ Auto-configured dark theme for better UX
- ✅ All dependencies are version-pinned for stability

### **Quick Deploy:**
1. Push to GitHub/GitLab
2. Connect to Streamlit Community Edition
3. Add API keys to secrets
4. Deploy! 🚀