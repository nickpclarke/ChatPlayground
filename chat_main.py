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

# --- Caching System Prompt ---
# @st.cache_data # Removed caching for system prompt to ensure fresh load
def load_system_prompt():
    SYSTEM_PROMPT_FILE = "system_prompt.txt"
    DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise."
    try:
        with open(SYSTEM_PROMPT_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        # Create the file with default content if it doesn't exist
        with open(SYSTEM_PROMPT_FILE, "w") as f:
            f.write(DEFAULT_SYSTEM_PROMPT)
        return DEFAULT_SYSTEM_PROMPT

# --- Page Configuration ---
st.set_page_config(
    page_title="🧠 Cerebras Chat",
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
        st.session_state.model = st.selectbox(
            "Model to Use",
            options=ALL_AVAILABLE_MODELS,
            index=default_model_index,
            key="unified_model_selector",
            help="Choose a model. Hover over options for details if descriptions are long."
        )
        # Model Descriptions with Tooltips (Point 2) - simplified display
        selected_model_desc = MODEL_DESCRIPTIONS.get(st.session_state.model, "No description available.")
        st.caption(f"**Using**: {st.session_state.model} - {selected_model_desc}")
    else:
        st.error("No chat models available for selection.")
        st.stop()

    # Advanced Toggle
    if st.toggle("🔧 Advanced Settings", key="advanced_settings_toggle"):
        st.session_state.max_completion_tokens = st.slider(
            "Max Completion Tokens", 
            100, 10000, 
            st.session_state.get("max_completion_tokens", 500), 
            key="max_completion_tokens_slider"
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

    # System Prompt Editor
    st.subheader("📝 System Prompt")
    # Ensure system_prompt is in session_state before text_area tries to access it
    if 'system_prompt' not in st.session_state:
        st.session_state.system_prompt = load_system_prompt()

    st.session_state.system_prompt = st.text_area(
        "System Instructions",
        value=st.session_state.system_prompt, # Use session state value
        height=150,
        placeholder="Example: 'You are a concise, detail-oriented AI assistant.'",
        key="system_editor"
    )
    if st.button("💾 Save Prompt", key="save_prompt_button"):
        try:
            with open("system_prompt.txt", "w") as f:
                f.write(st.session_state.system_prompt)
            st.toast("✅ System prompt saved!")
        except Exception as e:
            st.error(f"Error saving prompt: {e}")

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
                    if selected_model_name in CEREBRAS_MODEL_OPTIONS:
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
                    elif selected_model_name in GEMINI_MODEL_OPTIONS:
                        if gemini_api_configured: # Check if Gemini API was set up
                            st.toast(f"Calling Gemini model: {selected_model_name}...", icon="✨") # Debug Toast
                            from gemini_client import initialize_model # import only when needed
                            
                            # system_instruction_text will be extracted from messages_for_api
                            system_instruction_text_for_gemini = None
                            temp_messages_for_gemini = []
                            
                            # Convert messages to Gemini format [{role: "user"/"model", parts: [text]}]
                            # And extract system prompt
                            gemini_formatted_messages = []
                            for msg_idx, msg in enumerate(messages_for_api):
                                role = "model" if msg["role"] == "assistant" else msg["role"]
                                if role == "system" and msg_idx == 0: # System prompt should be first
                                    system_instruction_text_for_gemini = msg["content"]
                                    # Do not add system prompt to gemini_formatted_messages, it's passed to initialize_model
                                else:
                                    gemini_formatted_messages.append({"role": role, "parts": [msg["content"]]})
                            
                            gemini_model_instance = initialize_model(
                                selected_model_name, 
                                st, 
                                system_instruction=system_instruction_text_for_gemini
                            )

                            if gemini_model_instance:
                                gemini_generation_config = {}
                                if "max_completion_tokens" in api_params:
                                     gemini_generation_config["max_output_tokens"] = api_params["max_completion_tokens"]
                                if "temperature" in api_params:
                                     gemini_generation_config["temperature"] = api_params["temperature"]

                                response = gemini_model_instance.generate_content(
                                    contents=gemini_formatted_messages, # History without system prompt
                                    generation_config=genai.types.GenerationConfig(**gemini_generation_config) if gemini_generation_config else None,
                                )
                                raw_response = response.text
                                st.toast(f"Response received from Gemini: {selected_model_name}", icon="✅") # Debug Toast
                            else:
                                st.error(f"Failed to initialize Gemini model {selected_model_name} with system prompt '{system_instruction_text_for_gemini}'.")
                                st.stop()
                        else:
                            st.error("Gemini API not configured. Please set GEMINI_API_KEY.")
                            st.stop()
                    else:
                        st.error(f"Unknown model type for {selected_model_name}. Cannot proceed.")
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
                        with st.expander("🧠 Thought Process"):
                            st.caption(thought_content)
                    
                    st.markdown(content_with_anchor_and_link, unsafe_allow_html=True)
                    # Save the processed content (with HTML) to history
                    st.session_state.messages.append({"role": "assistant", "content": content_with_anchor_and_link})

                except Exception as e:
                    st.error(f"An error occurred with the Cerebras API: {e}")
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

    .stExpander {
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    html[data-theme="dark"] .stExpander {
        border: 1px solid #333333 !important;
    }

    .stExpander header {
        background-color: #f9f9f9 !important;
        padding: 8px 12px !important;
        border-radius: 7px 7px 0 0 !important;
    }
    html[data-theme="dark"] .stExpander header {
        background-color: #2c2c2c !important;
        color: #e0e0e0 !important;
    }

    .stExpander [data-testid="stExpanderDetails"] {
        background-color: #ffffff !important;
        padding: 10px !important;
        border-radius: 0 0 7px 7px !important;
    }
    html[data-theme="dark"] .stExpander [data-testid="stExpanderDetails"] {
        background-color: #1e1e1e !important;
        color: #d1d1d1 !important;
    }

    .stExpander [data-testid="stCaptionContainer"] {
        font-style: italic;
        color: #555;
        background-color: #f0f8ff;
        padding: 8px;
        border-radius: 4px;
    }
    html[data-theme="dark"] .stExpander [data-testid="stCaptionContainer"] {
        color: #aaa;
        background-color: #2a3950;
    }

    /* Sidebar styling based on app theme */
    [data-testid="stSidebar"] {
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }

    /* Light mode styling for sidebar */
    html[data-theme="light"] [data-testid="stSidebar"] {
        background-color: #ffffff !important; /* Light mode background */
        border-right: 1px solid #dee2e6 !important; /* Light mode border */
    }

    html[data-theme="light"] [data-testid="stSidebar"] .stSlider label,
    html[data-theme="light"] [data-testid="stSidebar"] .stSelectbox label,
    html[data-theme="light"] [data-testid="stSidebar"] .stTextInput label,
    html[data-theme="light"] [data-testid="stSidebar"] .stTextArea label {
        color: #333333 !important; /* Dark text for light mode */
    }

    /* Dark mode styling for sidebar */
    html[data-theme="dark"] [data-testid="stSidebar"] {
        background-color: #0E1117 !important; /* Streamlit's default dark sidebar */
        border-right: 1px solid #262730 !important; /* Streamlit's default dark border */
    }

    html[data-theme="dark"] [data-testid="stSidebar"] .stSlider label,
    html[data-theme="dark"] [data-testid="stSidebar"] .stSelectbox label,
    html[data-theme="dark"] [data-testid="stSidebar"] .stTextInput label,
    html[data-theme="dark"] [data-testid="stSidebar"] .stTextArea label {
        color: #fafafa !important; /* Light text for dark mode (Streamlit's default) */
    }

    /* Button styling for both modes */
    .stButton>button {
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out;
        border: 1px solid transparent;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    html[data-theme="dark"] .stButton>button:hover {
        box-shadow: 0 2px 4px rgba(255,255,255,0.1); /* Lighter shadow for dark mode */
    }

    /* Primary button */
    .stButton[kind="primary"]>button, .stButton>button[kind="primary"] {
        background-color: #007bff !important;
        color: white !important;
        border-color: #007bff !important;
    }
    .stButton[kind="primary"]>button:hover, .stButton>button[kind="primary"]:hover {
        background-color: #0056b3 !important;
        border-color: #0056b3 !important;
    }
    html[data-theme="dark"] .stButton[kind="primary"]>button, 
    html[data-theme="dark"] .stButton>button[kind="primary"] {
        background-color: #007bff !important; /* Keep primary color consistent or adjust for dark */
        border-color: #007bff !important;
    }
    html[data-theme="dark"] .stButton[kind="primary"]>button:hover, 
    html[data-theme="dark"] .stButton>button[kind="primary"]:hover {
        background-color: #0056b3 !important;
        border-color: #0056b3 !important;
    }


    /* Secondary button (like clear history) */
    .stButton[kind="secondary"]>button, .stButton>button[kind="secondary"] {
        background-color: #6c757d !important;
        color: white !important;
        border-color: #6c757d !important;
    }
    .stButton[kind="secondary"]>button:hover, .stButton>button[kind="secondary"]:hover {
        background-color: #545b62 !important;
        border-color: #545b62 !important;
    }
    html[data-theme="dark"] .stButton[kind="secondary"]>button,
    html[data-theme="dark"] .stButton>button[kind="secondary"] {
        background-color: #495057 !important; /* Darker secondary for dark mode */
        border-color: #495057 !important;
    }
    html[data-theme="dark"] .stButton[kind="secondary"]>button:hover,
    html[data-theme="dark"] .stButton>button[kind="secondary"]:hover {
        background-color: #3e444a !important;
        border-color: #3e444a !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)