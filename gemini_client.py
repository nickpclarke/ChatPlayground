import streamlit as st
import google.generativeai as genai
import google.genai as new_genai
from google.genai import types
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def configure_gemini_api(st_object_for_error_display):
    """Configures the Gemini API using the API key from Streamlit secrets or environment variables."""
    try:
        # Try to get API key from Streamlit secrets first
        api_key = None
        if hasattr(st, 'secrets') and "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        
        # Fall back to environment variables if not in secrets
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
            
        if not api_key:
            st_object_for_error_display.error("GEMINI_API_KEY not found in Streamlit secrets or environment variables. Please set it up.")
            return False
            
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st_object_for_error_display.error(f"Error configuring Gemini API: {e}")
        return False

def get_api_key():
    """Get the Gemini API key for the new client."""
    # Try Streamlit secrets first
    if hasattr(st, 'secrets') and "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    
    # Fall back to environment variables
    return os.getenv("GEMINI_API_KEY")

def is_thinking_model(model_name):
    """Check if a model supports thinking (Gemini 2.5 models)."""
    return "2.5" in model_name or "thinking" in model_name.lower()

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

def generate_content_with_thinking(model_name, messages, temperature=0.7, max_tokens=8192):
    """
    Generate content using Gemini with thinking support for 2.5 models.
    Returns tuple: (main_response, thinking_content)
    """
    try:
        api_key = get_api_key()
        if not api_key:
            return "Error: No API key available", None
        
        # Check if this is a thinking-capable model
        if not is_thinking_model(model_name):
            # Fall back to regular generation for non-thinking models
            return generate_content_regular(model_name, messages, temperature, max_tokens), None
        
        # Use new client for thinking models
        client = new_genai.Client(api_key=api_key)
        
        # Convert messages to content string
        content_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                content_parts.append(f"System: {content}")
            elif role == "user":
                content_parts.append(f"User: {content}")
            elif role == "assistant":
                content_parts.append(f"Assistant: {content}")
        
        full_content = "\n\n".join(content_parts)
        
        # Configure with thinking enabled
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(include_thoughts=True)
        )
        
        thinking_parts = []
        answer_parts = []
        
        # Try non-streaming first to avoid truncation issues
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=full_content,
                config=config
            )
            
            if not response:
                return "Error: Empty response from API", None
            
            # Process the complete response
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        for part in candidate.content.parts:
                            if not part:
                                continue
                                
                            # Check for thinking content
                            if hasattr(part, 'thought') and part.thought:
                                if hasattr(part, 'text') and part.text:
                                    thinking_parts.append(part.text)
                            else:
                                # Regular content
                                if hasattr(part, 'text') and part.text:
                                    answer_parts.append(part.text)
                
        except Exception as non_stream_error:
            # Fall back to streaming if non-streaming fails
            try:
                stream_response = client.models.generate_content_stream(
                    model=model_name,
                    contents=full_content,
                    config=config
                )
                
                if not stream_response:
                    return f"Error: Both streaming and non-streaming failed. Non-stream error: {str(non_stream_error)}", None
                
                for chunk in stream_response:
                    # Add comprehensive null checks
                    if not chunk:
                        continue
                        
                    if not hasattr(chunk, 'candidates') or not chunk.candidates:
                        continue
                        
                    # Check if candidates list is not empty and has content
                    if len(chunk.candidates) == 0:
                        continue
                        
                    candidate = chunk.candidates[0]
                    if not hasattr(candidate, 'content') or not candidate.content:
                        continue
                        
                    if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
                        continue
                        
                    # Now safely iterate over parts
                    for part in candidate.content.parts:
                        if not part:
                            continue
                            
                        # Check for thinking content
                        if hasattr(part, 'thought') and part.thought:
                            if hasattr(part, 'text') and part.text:
                                thinking_parts.append(part.text)
                        else:
                            # Regular content
                            if hasattr(part, 'text') and part.text:
                                answer_parts.append(part.text)
                                
            except Exception as stream_error:
                return f"Error: Both APIs failed. Non-stream: {str(non_stream_error)}, Stream: {str(stream_error)}", None
        
        # Combine results with better error handling
        main_response = "".join(answer_parts) if answer_parts else ""
        thinking_content = "\n\n".join(thinking_parts) if thinking_parts else None
        
        # Debug information (can be removed in production)
        total_chunks_processed = len(answer_parts) + len(thinking_parts)
        
        # Fallback if no content was captured
        if not main_response and not thinking_content:
            return f"Error: No content received from API (processed {total_chunks_processed} chunks)", None
            
        return main_response, thinking_content
        
    except Exception as e:
        # More detailed error information
        import traceback
        error_details = traceback.format_exc()
        return f"Error in generate_content_with_thinking: {str(e)}\nDetails: {error_details}", None

def generate_content_regular(model_name, messages, temperature=0.7, max_tokens=8192):
    """Generate content using regular Gemini API for non-thinking models."""
    try:
        # Convert messages to Gemini format
        gemini_messages = []
        system_instruction = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                role = "model" if msg["role"] == "assistant" else msg["role"]
                gemini_messages.append({"role": role, "parts": [msg["content"]]})
        
        # Initialize model with system instruction
        model_kwargs = {}
        if system_instruction:
            model_kwargs['system_instruction'] = system_instruction
        
        model = genai.GenerativeModel(model_name, **model_kwargs)
        
        # Generate content
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature
        )
        
        response = model.generate_content(
            contents=gemini_messages,
            generation_config=generation_config
        )
        
        return response.text
        
    except Exception as e:
        return f"Error: {str(e)}" 