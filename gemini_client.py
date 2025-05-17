import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def configure_gemini_api(st_object_for_error_display):
    """Configures the Gemini API using the API key from environment variables."""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st_object_for_error_display.error("GEMINI_API_KEY not found in environment variables. Please set it up.")
            return False
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st_object_for_error_display.error(f"Error configuring Gemini API: {e}")
        return False

def get_available_models(st_object_for_error_display):
    """Lists all available Gemini GenAI models suitable for content generation."""
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if not models:
            st_object_for_error_display.warning("No models supporting 'generateContent' found or API error.")
            return []
        return models
    except Exception as e:
        st_object_for_error_display.error(f"Error fetching models: {e}")
        return []

def initialize_model(model_name, st_object_for_error_display, system_instruction=None):
    """Initializes a Gemini GenerativeModel."""
    if not model_name:
        st_object_for_error_display.error("Model name not provided for initialization.")
        return None
    try:
        model_kwargs = {}
        if system_instruction:
            # Ensure system_instruction is in the correct format if necessary.
            # For basic text, it should be a string.
            # If using complex structures like genai.protos.Content, adjust accordingly.
            # For now, assume system_instruction is a simple string.
            model_kwargs['system_instruction'] = system_instruction

        model = genai.GenerativeModel(model_name, **model_kwargs)
        return model
    except Exception as e:
        st_object_for_error_display.error(f"Error initializing Gemini model ('{model_name}'): {e}")
        return None 