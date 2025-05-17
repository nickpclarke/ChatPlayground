import streamlit as st
import os
import re
import json
from io import StringIO

# --- IMMEDIATE API KEY CHECK ---
from cerebras_client import CerebrasClient, DEFAULT_MODEL

# --- Caching System Prompt ---
@st.cache_data
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
st.title("🤖 Cerebras Powered Chat")
st.write("A sleek, fast, and user-friendly chat UI using Cerebras.") # Genericized model name

# --- Model Options and Descriptions ---
# DEFAULT_MODEL is imported from cerebras_client
CEREBRAS_MODEL_OPTIONS = [
    "qwen-3-32b",
    "llama-4-scout-17b-16e-instruct",
    "llama3.1-8b",
    "llama-3.3-70b"
]
MODEL_DESCRIPTIONS = {
    "qwen-3-32b": "Fast inference, great for rapid iteration.",
    "llama-4-scout-17b-16e-instruct": "Optimized for guided workflows.",
    "llama3.1-8b": "Light and fast for quick tasks.",
    "llama-3.3-70b": "Most capable for complex reasoning."
}
# Ensure DEFAULT_MODEL is in options, if not, add it or pick a sensible default
if DEFAULT_MODEL not in CEREBRAS_MODEL_OPTIONS and CEREBRAS_MODEL_OPTIONS:
    # If DEFAULT_MODEL from SDK is valid but not in our hardcoded list,
    # we could add it, but for now, we prioritize the curated list.
    # If it's truly unavailable, the selection logic below will handle it.
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
    if current_model_in_session not in CEREBRAS_MODEL_OPTIONS:
        st.warning(f"Previously selected model '{current_model_in_session}' is no longer available. Reverting to default.")
        current_model_in_session = DEFAULT_MODEL
        if DEFAULT_MODEL not in CEREBRAS_MODEL_OPTIONS and CEREBRAS_MODEL_OPTIONS: # If default also not in list
            current_model_in_session = CEREBRAS_MODEL_OPTIONS[0]
        st.session_state.model = current_model_in_session
    
    # Determine default index robustly
    default_model_index = 0
    if CEREBRAS_MODEL_OPTIONS: # Ensure list is not empty
        try:
            default_model_index = CEREBRAS_MODEL_OPTIONS.index(current_model_in_session)
        except ValueError:
            default_model_index = 0 # Fallback to first model if current_model_in_session is somehow still not in the list

    if "model" not in st.session_state: # Initialize if not set
        st.session_state.model = CEREBRAS_MODEL_OPTIONS[default_model_index] if CEREBRAS_MODEL_OPTIONS else None

    if CEREBRAS_MODEL_OPTIONS:
        st.session_state.model = st.selectbox(
            "Model to Use",
            options=CEREBRAS_MODEL_OPTIONS,
            index=default_model_index,
            key="cerebras_model_selector",
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
            "Temperature (Creativity vs. Focus)",
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
    uploaded_file = st.file_uploader("Upload Chat History (JSON)", type=["json"], key="upload_chat_button")
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
                    # Clear the file uploader state to prevent re-processing on rerun
                    if "upload_chat_button" in st.session_state:
                        st.session_state.upload_chat_button = None # Set the widget's value to None
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
    api_key = None
    if hasattr(st, 'secrets') and "CEREBRAS_API_KEY" in st.secrets:
        api_key = st.secrets["CEREBRAS_API_KEY"]
    if not api_key:
        api_key = os.getenv("CEREBRAS_API_KEY")

    if not api_key:
        st.error("CEREBRAS_API_KEY not found. Please set it in .streamlit/secrets.toml or as an environment variable.")
        st.info("If you've just set it, please restart your terminal/IDE and Streamlit.")
        st.stop()
    
    try:
        client = CerebrasClient(api_key=api_key)
        return client
    except ValueError as ve: # Specific error from CerebrasClient if key is None (should be caught above)
        st.error(f"Cerebras Client Error: {ve}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to initialize Cerebras SDK: {e}")
        st.info("This might be due to an invalid API key, network issues, or Cerebras service status.")
        st.stop()

client = get_client() # Initialize client, will stop app if fails

# --- Model Descriptions (Optional, can be used later e.g. in tooltips) ---
# Moved MODEL_DESCRIPTIONS near CEREBRAS_MODEL_OPTIONS for clarity

# --- Chat History Setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Enhanced Message Rendering ---
for i, msg in enumerate(st.session_state.messages):
    avatar_icon = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])
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
            with st.spinner("Cerebras is thinking..."):
                try:
                    # Prepare message list
                    messages_for_api = []
                    current_system_prompt = st.session_state.get("system_prompt", load_system_prompt())
                    if current_system_prompt:
                        messages_for_api.append({"role": "system", "content": current_system_prompt})
                    
                    messages_for_api.extend(st.session_state.messages)
                    
                    api_params = {
                        "messages": messages_for_api,
                        "model": st.session_state.model
                    }
                    if "max_completion_tokens" in st.session_state and st.session_state.get("advanced_settings_toggle", False):
                        api_params["max_completion_tokens"] = st.session_state.max_completion_tokens
                    
                    # Add temperature if advanced settings are on (Point 7)
                    if st.session_state.get("advanced_settings_toggle", False):
                        api_params["temperature"] = st.session_state.temperature
                    
                    client_response = client.get_chat_completion(**api_params)
                    raw_response = client_response.choices[0].message.content
                    
                    thought_content = None
                    display_text = raw_response

                    think_match = re.search(r'< *think *>(.*?)< */ *think *>', raw_response, re.DOTALL | re.IGNORECASE)
                    
                    if think_match:
                        thought_content = think_match.group(1).strip()
                        before_think = raw_response[:think_match.start(0)]
                        after_think = raw_response[think_match.end(0):]
                        display_text = (before_think + after_think).strip()
                    
                    # Enhanced Error Handling for Large Responses (Point 9)
                    if len(display_text) > 20000: # Increased limit significantly
                        st.warning("Response is very long and might be slow to render fully.")
                        # Not truncating by default to preserve information, but warning is good.
                        # Consider truncation if performance becomes an issue:
                        # display_text = display_text[:15000] + "... (truncated for brevity)"

                    if thought_content:
                        with st.expander("🧠 Thought Process"):
                            st.caption(thought_content)
                    
                    st.markdown(display_text)
                    st.session_state.messages.append({"role": "assistant", "content": display_text})

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