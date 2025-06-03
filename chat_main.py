import streamlit as st
import os
import re
import json
from io import StringIO
import google.generativeai as genai

# Initialize session state for dynamic uploader key suffix
if 'uploader_key_suffix' not in st.session_state:
    st.session_state.uploader_key_suffix = 0

# --- Flag for clearing file uploader (This whole block can be removed now) ---
# if st.session_state.get("clear_upload_chat_button_flag", False):
#     if "upload_chat_button" in st.session_state: 
#         st.session_state.upload_chat_button = None
#     st.session_state.clear_upload_chat_button_flag = False

# --- IMMEDIATE API KEY CHECK ---
from cerebras_client import CerebrasClient, DEFAULT_MODEL as DEFAULT_CEREBRAS_MODEL
from gemini_client import configure_gemini_api, get_available_models as get_gemini_models

# --- System Prompts Management ---
def load_system_prompts():
    SYSTEM_PROMPTS_FILE = "system_prompts.json"
    DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise."
    
    try:
        with open(SYSTEM_PROMPTS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Create the file with default content if it doesn't exist or is invalid
        default_data = {
            "prompts": [
                {
                    "name": "Default Assistant",
                    "content": DEFAULT_SYSTEM_PROMPT
                }
            ],
            "selected": "Default Assistant"
        }
        with open(SYSTEM_PROMPTS_FILE, "w") as f:
            json.dump(default_data, f, indent=2)
        return default_data

def save_system_prompts(prompts_data):
    SYSTEM_PROMPTS_FILE = "system_prompts.json"
    try:
        with open(SYSTEM_PROMPTS_FILE, "w") as f:
            json.dump(prompts_data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving prompts: {e}")
        return False

# Legacy function for backward compatibility
def load_system_prompt():
    system_prompts = load_system_prompts()
    selected = system_prompts.get("selected")
    
    # Find the selected prompt
    for prompt in system_prompts.get("prompts", []):
        if prompt.get("name") == selected:
            return prompt.get("content")
    
    # Fallback to first prompt if selected not found
    if system_prompts.get("prompts"):
        return system_prompts["prompts"][0]["content"]
    
    # Ultimate fallback
    return "You are a helpful AI assistant. Be concise."

# --- Page Configuration ---
st.set_page_config(
    page_title="🧠 Unified Chat",
    page_icon="🤖",
    layout="centered"
)
st.title("🤖 Unified Model Chat")
st.write("A sleek, fast, and user-friendly chat UI using Cerebras and Gemini models.") # Genericized model name

# --- Model Options and Descriptions ---
# DEFAULT_MODEL is imported from cerebras_client
CEREBRAS_MODEL_OPTIONS = [
    "qwen-3-32b",
    "llama-4-scout-17b-16e-instruct",
    "llama3.1-8b",
    "llama-3.3-70b"
]
CEREBRAS_MODEL_DESCRIPTIONS = {
    "qwen-3-32b": "Cerebras: Fast inference, great for rapid iteration.",
    "llama-4-scout-17b-16e-instruct": "Cerebras: Optimized for guided workflows.",
    "llama3.1-8b": "Cerebras: Light and fast for quick tasks.",
    "llama-3.3-70b": "Cerebras: Most capable for complex reasoning."
}

# Configure Gemini API early
gemini_api_configured = configure_gemini_api(st)

# Fetch Gemini Models
GEMINI_MODEL_OPTIONS = []
if gemini_api_configured:
    GEMINI_MODEL_OPTIONS = get_gemini_models(st)

# Combine Model Lists
ALL_AVAILABLE_MODELS = CEREBRAS_MODEL_OPTIONS + GEMINI_MODEL_OPTIONS
MODEL_DESCRIPTIONS = {**CEREBRAS_MODEL_DESCRIPTIONS}
for gem_model in GEMINI_MODEL_OPTIONS:
    if gem_model not in MODEL_DESCRIPTIONS: # Avoid overwriting if a Gemini model happens to have the same name
        # Check if this is a thinking-capable model
        if "2.5" in gem_model or "thinking" in gem_model.lower():
            MODEL_DESCRIPTIONS[gem_model] = "Gemini: Advanced model with thinking capabilities 🧠"
        else:
            MODEL_DESCRIPTIONS[gem_model] = "Gemini: General purpose model."

# Determine a sensible overall default model
# Prioritize Cerebras default, then first Cerebras, then first Gemini, then None
DEFAULT_MODEL = DEFAULT_CEREBRAS_MODEL
if DEFAULT_MODEL not in ALL_AVAILABLE_MODELS:
    if CEREBRAS_MODEL_OPTIONS:
        DEFAULT_MODEL = CEREBRAS_MODEL_OPTIONS[0]
    elif GEMINI_MODEL_OPTIONS:
        DEFAULT_MODEL = GEMINI_MODEL_OPTIONS[0]
    else:
        DEFAULT_MODEL = None # No models available

# Ensure DEFAULT_MODEL is in options, if not, add it or pick a sensible default
if DEFAULT_MODEL is not None and DEFAULT_MODEL not in ALL_AVAILABLE_MODELS and ALL_AVAILABLE_MODELS:
    # This case should ideally not be hit if DEFAULT_MODEL logic above is correct
    pass

# --- Session State Initialization for new settings ---
if "max_history" not in st.session_state:
    st.session_state.max_history = 50 # Default max messages
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7 # Default temperature

# Dynamically update MAX_MESSAGES based on session state
MAX_MESSAGES = st.session_state.get("max_history", 50)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Settings")
    # Removed custom red styling for label as it can be distracting and might not be universally desired.
    # st.markdown("<style>div.stSelectbox>label{color:red;}</style>", unsafe_allow_html=True)

    # Model Selection
    # Refactored model selection logic (Point 8)
    current_model_in_session = st.session_state.get("model", DEFAULT_MODEL)
    if current_model_in_session not in ALL_AVAILABLE_MODELS:
        st.warning(f"Previously selected model '{current_model_in_session}' is no longer available. Reverting to default.")
        current_model_in_session = DEFAULT_MODEL
        if DEFAULT_MODEL not in ALL_AVAILABLE_MODELS and ALL_AVAILABLE_MODELS: # If default also not in list
            current_model_in_session = ALL_AVAILABLE_MODELS[0]
        st.session_state.model = current_model_in_session
    
    # Determine default index robustly
    default_model_index = 0
    if ALL_AVAILABLE_MODELS: # Ensure list is not empty
        try:
            default_model_index = ALL_AVAILABLE_MODELS.index(current_model_in_session)
        except ValueError:
            default_model_index = 0 # Fallback to first model if current_model_in_session is somehow still not in the list

    if "model" not in st.session_state: # Initialize if not set
        st.session_state.model = ALL_AVAILABLE_MODELS[default_model_index] if ALL_AVAILABLE_MODELS else None

    if ALL_AVAILABLE_MODELS:
        # Separate thinking and non-thinking models for better organization
        thinking_models = []
        regular_models = []
        
        for model in ALL_AVAILABLE_MODELS:
            if model in GEMINI_MODEL_OPTIONS and ("2.5" in model or "thinking" in model.lower()):
                thinking_models.append(model)
            elif model in CEREBRAS_MODEL_OPTIONS and any(keyword in model.lower() for keyword in ["qwen", "scout"]):
                thinking_models.append(model)
            else:
                regular_models.append(model)
        
        # Create organized model list with separators
        organized_models = []
        if thinking_models:
            organized_models.extend(thinking_models)
            if regular_models:
                organized_models.append("--- Standard Models ---")
                organized_models.extend(regular_models)
        else:
            organized_models = regular_models
        
        # Find current selection index in organized list
        if st.session_state.model in organized_models:
            default_model_index = organized_models.index(st.session_state.model)
        else:
            default_model_index = 0
        
        selected_model = st.selectbox(
            "Model to Use",
            options=organized_models,
            index=default_model_index,
            key="unified_model_selector",
            help="🧠 = Thinking-capable models that show reasoning process",
            format_func=lambda x: f"🧠 {x}" if x in thinking_models else x if x.startswith("---") else f"⚡ {x}"
        )
        
        # Skip separator selection
        if selected_model.startswith("---"):
            st.warning("Please select a model, not a separator.")
            st.stop()
        
        st.session_state.model = selected_model
        
        # Enhanced model descriptions with performance indicators
        selected_model_desc = MODEL_DESCRIPTIONS.get(st.session_state.model, "No description available.")
        
        # Add performance and capability indicators
        is_thinking_capable = selected_model in thinking_models
        model_type = "Cerebras" if selected_model in CEREBRAS_MODEL_OPTIONS else "Gemini"
        
        # Performance indicators
        performance_indicators = []
        if "qwen" in selected_model.lower():
            performance_indicators.extend(["🚀 Fast inference", "🧠 Structured thinking"])
        elif "scout" in selected_model.lower():
            performance_indicators.extend(["🎯 Guided workflows", "🧠 Step-by-step reasoning"])
        elif "2.5" in selected_model and "flash" in selected_model.lower():
            performance_indicators.extend(["⚡ Ultra-fast", "🧠 Advanced thinking", "💰 Cost-efficient"])
        elif "2.5" in selected_model and "pro" in selected_model.lower():
            performance_indicators.extend(["🧠 Deep reasoning", "📊 Complex analysis", "🎯 High accuracy"])
        elif "flash" in selected_model.lower():
            performance_indicators.extend(["⚡ Fast responses", "💰 Budget-friendly"])
        elif "pro" in selected_model.lower():
            performance_indicators.extend(["🎯 High quality", "📊 Detailed analysis"])
        
        # Display enhanced model info
        st.caption(f"**Using**: {selected_model}")
        st.caption(f"**Type**: {model_type} | **Thinking**: {'✅ Enabled' if is_thinking_capable else '❌ Not available'}")
        if performance_indicators:
            st.caption(f"**Strengths**: {' | '.join(performance_indicators)}")
        
        # Show thinking status prominently for thinking models
        if is_thinking_capable:
            st.success("🧠 This model will show its reasoning process!")
        
    else:
        st.error("No chat models available for selection.")
        st.stop()

    # Advanced Toggle
    if st.toggle("🔧 Advanced Settings", key="advanced_settings_toggle"):
        # Determine intelligent default for max completion tokens based on model type
        current_model = st.session_state.get("model", "")
        is_thinking_model_selected = (current_model in GEMINI_MODEL_OPTIONS and ("2.5" in current_model or "thinking" in current_model.lower())) or \
                                   (current_model in CEREBRAS_MODEL_OPTIONS and any(keyword in current_model.lower() for keyword in ["qwen", "scout"]))
        
        # Set higher default for thinking models
        intelligent_default = 4000 if is_thinking_model_selected else 1500
        
        # Initialize with intelligent default if not already set
        if "max_completion_tokens" not in st.session_state:
            st.session_state.max_completion_tokens = intelligent_default
        
        st.session_state.max_completion_tokens = st.slider(
            "Max Completion Tokens", 
            100, 10000, 
            st.session_state.get("max_completion_tokens", intelligent_default), 
            key="max_completion_tokens_slider",
            help=f"💡 {'🧠 Thinking models need more tokens!' if is_thinking_model_selected else 'Standard models work well with fewer tokens'}"
        )
        # Configurable Max Message Limit (Point 3)
        st.session_state.max_history = st.slider(
            "Max Messages in History",
            10, 200, # Adjusted max for practicality
            st.session_state.get("max_history", 50),
            key="history_limit_slider"
        )
        MAX_MESSAGES = st.session_state.max_history # Update global MAX_MESSAGES

        # Add Temperature Slider (Point 7)
        st.session_state.temperature = st.slider(
            "Temperature (Focus vs. Creativity)",
            0.0, 1.5, # Range from API docs
            st.session_state.get("temperature", 0.7), # Default 0.7
            step=0.05,
            key="temperature_slider",
            help="Lower values (e.g., 0.2) are more deterministic; higher values (e.g., 0.8) are more random."
        )
        
        # Thinking Settings Panel
        st.markdown("---")
        st.subheader("🧠 Thinking Settings")
        
        # Only show thinking settings if a thinking model is selected
        current_model = st.session_state.get("model", "")
        is_current_thinking = (current_model in GEMINI_MODEL_OPTIONS and ("2.5" in current_model or "thinking" in current_model.lower())) or \
                             (current_model in CEREBRAS_MODEL_OPTIONS and any(keyword in current_model.lower() for keyword in ["qwen", "scout"]))
        
        if is_current_thinking:
            # Thinking display preferences
            st.session_state.thinking_auto_expand = st.checkbox(
                "Auto-expand thinking process",
                value=st.session_state.get("thinking_auto_expand", False),
                help="Automatically expand the thinking section for new responses"
            )
            
            st.session_state.thinking_show_stats = st.checkbox(
                "Show thinking statistics",
                value=st.session_state.get("thinking_show_stats", True),
                help="Display thinking chunk count and processing info"
            )
            
            # Thinking verbosity for Gemini models
            if current_model in GEMINI_MODEL_OPTIONS:
                st.session_state.thinking_verbosity = st.selectbox(
                    "Thinking Detail Level",
                    options=["Concise", "Normal", "Detailed"],
                    index=1,  # Default to Normal
                    key="thinking_verbosity_selector",
                    help="Control how much thinking detail the model should show"
                )
            
            st.info("💡 Thinking models will show their reasoning process in an expandable section below responses.")
        else:
            st.info("🔄 Select a thinking-capable model (🧠) to access thinking settings.")

    # System Prompt Management
    st.subheader("📝 System Prompts")
    
    # Load system prompts
    system_prompts_data = load_system_prompts()
    prompts_list = system_prompts_data.get("prompts", [])
    selected_prompt = system_prompts_data.get("selected")
    
    # Get prompt names for selection
    prompt_names = [prompt["name"] for prompt in prompts_list]
    
    # Select prompt
    if prompt_names:
        selected_index = 0
        if selected_prompt in prompt_names:
            selected_index = prompt_names.index(selected_prompt)
        
        selected_name = st.selectbox(
            "Select System Prompt",
            options=prompt_names,
            index=selected_index,
            key="system_prompt_selector"
        )
        
        # Update selected prompt in session state and data
        system_prompts_data["selected"] = selected_name
        
        # Find the content of selected prompt
        selected_content = ""
        for prompt in prompts_list:
            if prompt["name"] == selected_name:
                selected_content = prompt["content"]
                break
        
        # Set system prompt in session state
        if 'system_prompt' not in st.session_state:
            st.session_state.system_prompt = selected_content
        elif selected_name != st.session_state.get("last_selected_prompt"):
            st.session_state.system_prompt = selected_content
        
        # Remember last selected prompt
        st.session_state.last_selected_prompt = selected_name
        
        # Display and edit the selected prompt
        new_content = st.text_area(
            "Edit System Instructions",
            value=st.session_state.system_prompt,
            height=150,
            placeholder="Example: 'You are a concise, detail-oriented AI assistant.'",
            key="system_editor"
        )
        
        # Update session state with edited content
        st.session_state.system_prompt = new_content
        
        # Save button for the current prompt
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Update Prompt", key="update_prompt_button"):
                # Update the prompt content
                for prompt in prompts_list:
                    if prompt["name"] == selected_name:
                        prompt["content"] = new_content
                        break
                
                # Save changes
                if save_system_prompts(system_prompts_data):
                    st.toast(f"✅ System prompt '{selected_name}' updated!")
        
        with col2:
            if st.button("❌ Delete Prompt", key="delete_prompt_button", 
                        disabled=len(prompts_list) <= 1):  # Prevent deleting the last prompt
                # Remove the prompt
                prompts_list = [p for p in prompts_list if p["name"] != selected_name]
                system_prompts_data["prompts"] = prompts_list
                
                # Update selected to first available prompt if deleted the selected one
                if selected_name == selected_prompt and prompts_list:
                    system_prompts_data["selected"] = prompts_list[0]["name"]
                
                # Save changes
                if save_system_prompts(system_prompts_data):
                    st.toast(f"✅ System prompt '{selected_name}' deleted!")
                    st.rerun()
    
    # New prompt section
    # Use an expander instead of checkbox and subheader
    
    # Initialize form key suffix if it doesn't exist (for generating new keys)
    if "form_key_suffix" not in st.session_state:
        st.session_state.form_key_suffix = 0
        
    # Generate current keys for form elements
    name_key = f"new_prompt_name_{st.session_state.form_key_suffix}"
    content_key = f"new_prompt_content_{st.session_state.form_key_suffix}"
    
    # Always set expanded=False to keep it closed by default
    with st.expander("➕ Add New Prompt", expanded=False):
        new_prompt_name = st.text_input("New Prompt Name", key=name_key)
        new_prompt_content = st.text_area(
            "New Prompt Content", 
            height=100,
            placeholder="Enter instructions for the AI here...",
            key=content_key
        )
        
        # Regular button, no form
        if st.button("➕ Add Prompt", key="add_prompt_button", 
                    disabled=not new_prompt_name or not new_prompt_content):
            # Check if name already exists
            if new_prompt_name in prompt_names:
                st.error(f"A prompt with name '{new_prompt_name}' already exists.")
            else:
                # Add new prompt
                prompts_list.append({
                    "name": new_prompt_name,
                    "content": new_prompt_content
                })
                system_prompts_data["prompts"] = prompts_list
                
                # Save changes
                if save_system_prompts(system_prompts_data):
                    # Increment the form key suffix to get new empty fields on next render
                    st.session_state.form_key_suffix += 1
                    st.toast(f"✅ New system prompt '{new_prompt_name}' added!")
                    st.rerun()

    # Chat Actions
    st.subheader("🗑️ Actions")
    if st.button("❌ Clear Chat History", key="clear_chat_button"):
        st.session_state.messages = []
        st.toast("🧹 Chat history cleared!")
        st.rerun()

    # Load/Download Chat
    st.subheader("📁 File Actions")
    # Use a dynamic key for the file uploader
    uploader_instance_key = f"upload_chat_button_{st.session_state.uploader_key_suffix}"
    uploaded_file = st.file_uploader(
        "Upload Chat History (JSON)", 
        type=["json"], 
        key=uploader_instance_key
    )
    if uploaded_file:
        try:
            uploaded_content = uploaded_file.read().decode('utf-8') # Explicitly specify utf-8
            if not uploaded_content.strip(): # Check for empty or whitespace-only content
                st.error("Uploaded JSON file is empty.")
            else:
                parsed_json = json.loads(uploaded_content)
                
                # Enhanced validation: Check structure and types
                valid_format = False
                if parsed_json is not None and isinstance(parsed_json, list):
                    # Check if all items in the list are dicts with correct keys and string values
                    valid_format = all(
                        isinstance(item, dict) and \
                        isinstance(item.get("role"), str) and \
                        isinstance(item.get("content"), str)
                        for item in parsed_json
                    )

                if valid_format:
                    st.session_state.messages = parsed_json
                    st.toast("✅ Chat history loaded!")
                    # Increment the key suffix to reset the uploader on the next run
                    st.session_state.uploader_key_suffix += 1
                    # st.session_state.clear_upload_chat_button_flag = True # Old flag logic, remove
                    st.rerun()
                else:
                    st.error("Invalid JSON format. Expected a list of objects, each with a string 'role' and string 'content'. E.g., [{\"role\": \"user\", \"content\": \"Hi\"}, ...]")
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Could not decode JSON content. Please ensure correct JSON syntax.")
        except UnicodeDecodeError:
            st.error("Error decoding file. Please ensure it's a valid UTF-8 encoded JSON file.")
        except Exception as e:
            st.error(f"An unexpected error occurred while loading chat: {e}")

    if "messages" in st.session_state and st.session_state.messages:
        try:
            chat_export_data = json.dumps(st.session_state.messages, indent=2)
            st.download_button(
                label="⬇️ Download Chat History (JSON)",
                data=chat_export_data,
                file_name="chat_history.json",
                mime="application/json",
                key="download_chat_button"
            )
        except Exception as e:
            st.error(f"Error preparing download: {e}")
            st.stop()

# --- Cerebras Client Setup (Cached) ---
@st.cache_resource
def get_client():
    # Try to get API key from st.secrets first, then environment variable
    # CEREBRAS API KEY Handling
    cerebras_api_key = None
    if hasattr(st, 'secrets') and "CEREBRAS_API_KEY" in st.secrets:
        cerebras_api_key = st.secrets["CEREBRAS_API_KEY"]
    if not cerebras_api_key:
        cerebras_api_key = os.getenv("CEREBRAS_API_KEY")

    # GEMINI API KEY is handled by gemini_client.py using os.getenv("GEMINI_API_KEY")
    # We've already called configure_gemini_api() which uses st.error for feedback.

    # Initialize Cerebras client if key is present
    cerebras_client_instance = None
    if cerebras_api_key:
        try:
            cerebras_client_instance = CerebrasClient(api_key=cerebras_api_key)
        except ValueError as ve:
            st.warning(f"Cerebras Client Error: {ve}. Cerebras models may not be available.")
        except Exception as e:
            st.warning(f"Failed to initialize Cerebras SDK: {e}. Cerebras models may not be available.")
    else:
        st.info("CEREBRAS_API_KEY not found or not set. Cerebras models will not be available. Set it in .streamlit/secrets.toml or as an environment variable if you wish to use them.")
        # If only Cerebras models were listed and no key, this is a bigger issue.
        # But now we have Gemini as a potential fallback.

    # The get_client function now needs to potentially return multiple clients or a dispatcher
    # For now, we'll handle client selection in the chat generation logic.
    # This function's role is more about ensuring API keys are checked and base clients are attempted.
    # The actual Gemini model object is initialized later, on demand.
    return {"cerebras": cerebras_client_instance} # Return a dict of clients

client_connections = get_client() # Initialize client connections

# --- Model Descriptions (Optional, can be used later e.g. in tooltips) ---
# Moved MODEL_DESCRIPTIONS near CEREBRAS_MODEL_OPTIONS for clarity

# --- Chat History Setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Enhanced Message Rendering ---
for i, msg in enumerate(st.session_state.messages):
    avatar_icon = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        # Assistant messages might contain HTML for anchor and link if they were long.
        # User messages do not. unsafe_allow_html is safe for both.
        st.markdown(msg["content"], unsafe_allow_html=True)
    # Add a divider only if it's not the last message AND there are more messages after it.
    # This prevents a divider after the very last message.
    if i < len(st.session_state.messages) - 1 : 
        st.markdown("---")

# --- Chat Input Handler ---
if prompt := st.chat_input("Send a message..."):
    # Handle Empty User Prompts (Point 4)
    stripped_prompt = prompt.strip()
    if not stripped_prompt:
        st.toast("💬 Please enter a non-empty message.")
    else:
        st.session_state.messages.append({"role": "user", "content": stripped_prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(stripped_prompt)

        # Generate with error boundary
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("The AI is thinking..."):
                try:
                    # Prepare message list
                    messages_for_api = []
                    current_system_prompt = st.session_state.get("system_prompt", load_system_prompt())
                    if current_system_prompt:
                        messages_for_api.append({"role": "system", "content": current_system_prompt})
                    
                    # Ensure user messages are included; system prompt handled above
                    # Filter out any existing system messages from st.session_state.messages to avoid duplication
                    history_for_api = [msg for msg in st.session_state.messages if msg["role"] != "system"]
                    messages_for_api.extend(history_for_api)
                    
                    selected_model_name = st.session_state.model
                    api_params = {
                        "messages": messages_for_api,
                        "model": selected_model_name # This is just the name, client logic will use it
                    }
                    if "max_completion_tokens" in st.session_state and st.session_state.get("advanced_settings_toggle", False):
                        api_params["max_completion_tokens"] = st.session_state.max_completion_tokens
                    
                    # Add temperature if advanced settings are on (Point 7)
                    if st.session_state.get("advanced_settings_toggle", False):
                        api_params["temperature"] = st.session_state.temperature
                    
                    raw_response = ""
                    thought_content = None # Initialize thought_content

                    # --- Client Dispatch Logic ---
                    # Better model detection logic
                    is_cerebras_model = selected_model_name in CEREBRAS_MODEL_OPTIONS
                    is_gemini_model = selected_model_name in GEMINI_MODEL_OPTIONS or "gemini" in selected_model_name.lower()
                    
                    if is_cerebras_model:
                        if client_connections["cerebras"]:
                            st.toast(f"Calling Cerebras model: {selected_model_name}...", icon="🧠") # Debug Toast
                            # Cerebras specific params might need adjustment if different from Gemini
                            cerebras_api_params = {
                                "messages": api_params["messages"], # Pass the combined history + system
                                "model": selected_model_name
                            }
                            if "max_completion_tokens" in api_params:
                                cerebras_api_params["max_tokens"] = api_params["max_completion_tokens"] # Cerebras uses max_tokens
                            if "temperature" in api_params:
                                cerebras_api_params["temperature"] = api_params["temperature"]
                            
                            client_response = client_connections["cerebras"].get_chat_completion(**cerebras_api_params)
                            raw_response = client_response.choices[0].message.content
                            st.toast(f"Response received from Cerebras: {selected_model_name}", icon="✅") # Debug Toast
                        else:
                            st.error(f"Cerebras client not available for model {selected_model_name}. Check API key.")
                            st.stop()
                    elif is_gemini_model:
                        if gemini_api_configured: # Check if Gemini API was set up
                            st.toast(f"Calling Gemini model: {selected_model_name}...", icon="✨") # Debug Toast
                            from gemini_client import generate_content_with_thinking, is_thinking_model # import thinking functions
                            
                            # Extract temperature and max_tokens from api_params
                            temperature = api_params.get("temperature", 0.7)
                            max_tokens = api_params.get("max_completion_tokens", 8192)
                            
                            # Call the new thinking-enabled function
                            raw_response, thinking_content = generate_content_with_thinking(
                                model_name=selected_model_name,
                                messages=messages_for_api,
                                temperature=temperature,
                                max_tokens=max_tokens
                            )
                            
                            # Check if there was an error
                            if raw_response.startswith("Error:"):
                                st.error(f"Gemini API error: {raw_response}")
                                st.stop()
                            
                            # Set thought_content for thinking models
                            if thinking_content:
                                thought_content = thinking_content
                            
                            st.toast(f"Response received from Gemini: {selected_model_name}", icon="✅") # Debug Toast
                        else:
                            st.error("Gemini API not configured. Please set GEMINI_API_KEY.")
                            st.stop()
                    else:
                        st.error(f"Unknown model type for {selected_model_name}. This model is not recognized as either Cerebras or Gemini.")
                        st.stop()
                    # --- End Client Dispatch ---

                    display_text = raw_response
                    think_match = re.search(r'< *think *>(.*?)< */ *think *>', raw_response, re.DOTALL | re.IGNORECASE)
                    
                    if think_match:
                        thought_content = think_match.group(1).strip()
                        before_think = raw_response[:think_match.start(0)]
                        after_think = raw_response[think_match.end(0):]
                        display_text = (before_think + after_think).strip()
                    
                    # --- Add anchor and "Back to top" link for long assistant messages ---
                    LONG_MESSAGE_THRESHOLD = 750 # Characters
                    # The index of the new assistant message (after user's prompt and this one are added)
                    assistant_message_index = len(st.session_state.messages) 
                    anchor_id = f"msg-anchor-{assistant_message_index}"

                    # Prepare content with anchor at the beginning
                    # Using a div for the anchor ensures it's a block element and content flows after it.
                    content_with_anchor_and_link = f"<div id='{anchor_id}'></div>{display_text}"

                    if len(display_text) > LONG_MESSAGE_THRESHOLD:
                        back_to_top_link_html = (
                            f"<br><p style='text-align: right; font-size: 0.8em; margin-top: 10px; margin-bottom: 0px;'>"
                            f"<a href='#{anchor_id}'>⬆️ Back to top of this response</a>"
                            f"</p>"
                        )
                        content_with_anchor_and_link += back_to_top_link_html
                    # --- End of new logic ---

                    if thought_content:
                        # Use user settings for thinking display
                        auto_expand = st.session_state.get("thinking_auto_expand", False)
                        show_stats = st.session_state.get("thinking_show_stats", True)
                        
                        # Count thinking chunks/words for statistics
                        thinking_stats = ""
                        if show_stats:
                            word_count = len(thought_content.split())
                            char_count = len(thought_content)
                            thinking_stats = f" ({word_count} words, {char_count} characters)"
                        
                        with st.expander(f"🧠 Thought Process{thinking_stats}", expanded=auto_expand):
                            # Add thinking content with better formatting
                            st.markdown("**Model's reasoning process:**")
                            
                            # Process thinking content for better formatting
                            formatted_thinking = thought_content
                            
                            # Add proper line breaks for common patterns
                            formatted_thinking = formatted_thinking.replace('. ', '.\n\n')  # Paragraph breaks
                            formatted_thinking = formatted_thinking.replace('? ', '?\n\n')  # Question breaks
                            formatted_thinking = formatted_thinking.replace(': ', ':\n\n')  # Colon breaks
                            
                            # Handle numbered lists better
                            import re
                            formatted_thinking = re.sub(r'(\d+\.)\s*', r'\n\n**\1** ', formatted_thinking)
                            
                            # Handle bullet points
                            formatted_thinking = re.sub(r'([•\-\*])\s*', r'\n\n\1 ', formatted_thinking)
                            
                            # Clean up excessive line breaks
                            formatted_thinking = re.sub(r'\n{3,}', '\n\n', formatted_thinking)
                            formatted_thinking = formatted_thinking.strip()
                            
                            # Display as markdown in a container with custom styling
                            st.markdown(
                                f"""
                                <div class="thinking-content">
                                {formatted_thinking.replace(chr(10), '<br>')}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                    st.markdown(content_with_anchor_and_link, unsafe_allow_html=True)
                    # Save the processed content (with HTML) to history
                    st.session_state.messages.append({"role": "assistant", "content": content_with_anchor_and_link})

                except Exception as e:
                    # Determine which API we were using for better error context
                    api_type = "Gemini" if is_gemini_model else "Cerebras" if is_cerebras_model else "AI"
                    st.error(f"An error occurred with the {api_type} API: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Error: {e}"})

# --- Session Expire Handling (Hardcoded MAX_MESSAGES is now dynamic) ---
# Trim Old Messages with Feedback (Point 6)
if len(st.session_state.messages) > MAX_MESSAGES: # MAX_MESSAGES is now dynamic
    st.warning(f"♻️ Chat history trimmed to the latest {MAX_MESSAGES} messages (oldest ones removed).")
    st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]

# --- Debug Mode (Point 10) ---
if st.sidebar.toggle("🔍 Debug Mode", key="debug_mode_toggle", help="Show debug information like session state."):
    st.sidebar.subheader("🐛 Debug Info")
    st.sidebar.write("Session State:")
    st.sidebar.json(st.session_state)
    # Avoid printing client object directly if it's complex or sensitive.
    # Can show specific client attributes if needed.
    # st.sidebar.write("Cerebras Client Details (Example):")
    # st.sidebar.text(f"API Endpoint: {client.api_url}") # Assuming client has such an attribute

# --- Custom CSS ---
st.markdown(
    """
    <style>
    /* General chat message container styling */
    [data-testid="stChatMessage"] { 
        margin-bottom: 10px !important;
    }

    /* User message styling - targeting Streamlit's data-theme for specificity */
    html[data-theme="light"] [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent"][data-test-id="stChatMessageContent-user"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent"][data-test-id="stChatMessageContent-user"]) [data-testid="stChatMessageContent"] /* Fallback for user messages */ {
        background-color: #e6f2ff !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    html[data-theme="dark"] [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent"][data-test-id="stChatMessageContent-user"]) [data-testid="stChatMessageContent"] {
        background-color: #2a3950 !important; /* Darker blue for dark mode user messages */
        border-radius: 10px !important;
        padding: 10px !important;
        color: #e0e0e0 !important; /* Lighter text for dark user messages */
    }

    /* Assistant message styling - targeting Streamlit's data-theme for specificity */
    html[data-theme="light"] [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent"][data-test-id="stChatMessageContent-assistant"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent"][data-test-id="stChatMessageContent-assistant"]) [data-testid="stChatMessageContent"] /* Fallback for assistant messages */ {
        background-color: #f5f5f5 !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    html[data-theme="dark"] [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent"][data-test-id="stChatMessageContent-assistant"]) [data-testid="stChatMessageContent"] {
        background-color: #262730 !important; /* Darker grey for dark mode assistant messages */
        border-radius: 10px !important;
        padding: 10px !important;
        color: #d1d1d1 !important; /* Lighter text for dark assistant messages */
    }

    .stMarkdown [data-testid="stMarkdownContainer"] {
        white-space: pre-wrap !important;
    }

    /* Basic styling for expanders - simplified and fixed */
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-top: 5px;
        margin-bottom: 10px;
    }
    html[data-theme="dark"] .stExpander {
        border: 1px solid #3d3d3d;
    }

    /* Input styling - simplified */
    input[type="text"], textarea {
        border-radius: 6px;
        border: 1px solid #ced4da;
    }
    html[data-theme="dark"] input[type="text"], 
    html[data-theme="dark"] textarea {
        border: 1px solid #4d5154;
    }
    
    /* Button styling - simplified */
    .stButton button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.15s ease-in-out;
    }
    
    /* Primary buttons (Add, Update) */
    .stButton button {
        background-color: #4e7496;
        color: white;
    }
    .stButton button:hover:enabled {
        background-color: #3a5a78;
        transform: translateY(-1px);
    }
    
    /* Danger button (Delete) */
    button[kind="error"], button[data-testid="baseButton-secondary"] {
        background-color: #6c757d;
    }
    
    /* Fix layout spacing */
    .stExpander [data-testid="stExpanderDetails"] {
        padding: 10px;
    }
    
    /* Enhanced thinking section styling */
    .thinking-process {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 4px solid #007bff;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    html[data-theme="dark"] .thinking-process {
        background: linear-gradient(135deg, #2c2c2c 0%, #3c3c3c 100%);
        border-left: 4px solid #4dabf7;
    }
    
    /* Model capability indicators */
    .model-capability {
        display: inline-block;
        padding: 3px 8px;
        margin: 2px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: 500;
    }
    
    .thinking-capable {
        background-color: #e3f2fd;
        color: #1976d2;
        border: 1px solid #90caf9;
    }
    
    .fast-inference {
        background-color: #fff3e0;
        color: #f57c00;
        border: 1px solid #ffb74d;
    }
    
    html[data-theme="dark"] .thinking-capable {
        background-color: #1a2332;
        color: #90caf9;
        border: 1px solid #1976d2;
    }
    
    html[data-theme="dark"] .fast-inference {
        background-color: #2d1b0d;
        color: #ffb74d;
        border: 1px solid #f57c00;
    }
    
    /* Thinking statistics styling */
    .thinking-stats {
        font-size: 0.85em;
        color: #666;
        font-style: italic;
    }
    
    html[data-theme="dark"] .thinking-stats {
        color: #999;
    }
    
    /* Enhanced thinking content display */
    .thinking-content {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        max-height: 400px;
        overflow-y: auto;
        color: #333;
    }
    
    html[data-theme="dark"] .thinking-content {
        background-color: #2c2c2c;
        border-left: 4px solid #4dabf7;
        color: #e0e0e0;
    }

    </style>
    """,
    unsafe_allow_html=True
)