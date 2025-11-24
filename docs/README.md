# ReqVibe - AI Requirements Analyst

A Streamlit application that uses a centralized LLM gateway to orchestrate multiple OpenAI-compatible providers (DeepSeek, GPT-4o, Claude 3, Gemini 1.5, Grok, etc.) for requirement analysis.

## Character Card

**This is the DNA of ReqVibe — all behavior comes from here.**

The `config/roles/analyst.json` file (default) defines ReqVibe's personality, expertise, and interaction style. Multiple role configurations are available in `config/roles/`:
- `analyst.json` - Requirements analyst (default)
- `architect.json` - Software architect
- `developer.json` - Full-stack developer
- `tester.json` - QA engineer

These configuration files ensure consistent, professional requirements engineering assistance following industry standards like Volere and IEEE 830.

See `config/roles/analyst.json` for the default character definition.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set the required environment variables (see the section below for details)
3. Launch the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default web browser.

## Environment Variables

All configuration is read from a `.env` file in the project root (automatically loaded) or from your shell.

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `CENTRALIZED_LLM_API_KEY` | API key for the centralized LLM gateway that proxies DeepSeek/OpenAI-compatible providers | `CENTRALIZED_LLM_API_KEY=sk-xxxxx` |
| `UNSTRUCTURED_API_KEY` | API key for the Unstructured.io document processing service | `UNSTRUCTURED_API_KEY=uzzzzz` |

### Optional (enable only what you need)

| Variable | Description |
|----------|-------------|
| `UNSTRUCTURED_API_URL` | Custom Unstructured endpoint, defaults to `https://api.unstructured.io/general/v0/general` |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT` | Enable LangSmith observability |
| `VOICE_TRANSCRIBE_MODEL`, `VOICE_TRANSCRIBE_LANGUAGE`, `VOICE_TRANSCRIBE_TEMPERATURE` | Override default Whisper transcription settings |

### `.env` template

```
# LLM Gateway
CENTRALIZED_LLM_API_KEY=your_key

# Document processing
UNSTRUCTURED_API_KEY=your_key
# UNSTRUCTURED_API_URL=https://api.unstructured.io/general/v0/general

# Optional services
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=your_langsmith_key

# Voice transcription tweaks (optional)
# VOICE_TRANSCRIBE_MODEL=base
# VOICE_TRANSCRIBE_LANGUAGE=en
# VOICE_TRANSCRIBE_TEMPERATURE=0.1
```

> **Never commit the `.env` file.** It is already excluded by `.gitignore`.

To set variables manually without `.env`, use your shell’s syntax (`$env:VAR="value"` in PowerShell, `export VAR=value` in Bash/Zsh, etc.).

## Memory Architecture Snapshot

ReqVIBE uses a layered memory model optimised for Streamlit reruns:

1. **`ShortTermMemory` (`core/models/memory.py`)**  
   - Maintains the active chat transcript, extracted requirements, and token usage.  
   - Provides `get_context_for_api()` to enforce token budgets before hitting the LLM.

2. **Streamlit Session State (`utils/state_manager.py`)**  
   - Stores UI selections (role, model), authentication state, and current conversation metadata.  
   - Keeps an instance of `ShortTermMemory` per session to survive reruns without re-fetching data.

3. **Persistent Storage (`domain/conversations/service.py`)**  
   - Saves the last N conversations per user (JSON on disk) with per-user size limits.  
   - Uses signatures to avoid redundant writes and sorts sessions by `created_at` during truncation.

4. **Caching**  
   - Role cards and sidebar icons are cached using `st.cache_data` to avoid repeated disk IO.  
   - Whisper models are cached with `@st.cache_resource`, so the base model is loaded once per session.

This layered approach keeps the UI responsive while providing durability for authenticated users.

## Usage

1. Enter your requirement or question in the text input box
2. Click the "Ask" button
3. View the AI response below
4. Continue the conversation or clear it to start over

## Requirements

- Python 3.9+ (recommended)
- Centralized LLM API key (OpenAI-compatible; DeepSeek/Grok/GPT/Claude/Gemini are all supported)
- `streamlit`
- `openai` (client used for all OpenAI-compatible gateways)

## Security Note

⚠️ **Important:** Never commit your API key to version control. The API key in the setup script and README is for convenience only. In production, use environment variables or secure credential management systems.

## Streamlit Cloud Deployment

To deploy this app to Streamlit Cloud:

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Sign in with your GitHub account
4. Click "New app" and select your repository
5. Set the main file path to: `app.py`
6. Add your required secrets (at minimum `CENTRALIZED_LLM_API_KEY`, plus anything like `UNSTRUCTURED_API_KEY`)
7. Click "Deploy"

**Important:** Set secrets in Streamlit Cloud (not in the code), for example:
```
CENTRALIZED_LLM_API_KEY = "YOUR_GATEWAY_KEY"
UNSTRUCTURED_API_KEY = "YOUR_DOC_KEY"
```

For detailed deployment instructions, see [DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Features

- ✅ Chat history maintained in session state
- ✅ Volere template structure for requirements analysis
- ✅ Professional requirements engineering prompts
- ✅ Real-time API responses with loading indicators
- ✅ Clear conversation functionality
- ✅ Responsive UI with Streamlit chat interface
- ✅ IEEE 830 SRS document generation
- ✅ Context summarization for long conversations

## Extending with Other API Providers

This project is designed to work with any OpenAI-compatible API via the centralized gateway (`config/models.py`). Out of the box it ships with presets for DeepSeek, OpenAI, Anthropic Claude, Grok, and Google Gemini, and you can add more by editing the model registry.

### Architecture Overview

The project uses the OpenAI SDK pattern, which is compatible with many providers:
- **DeepSeek / Grok / Perplexity**: OpenAI-compatible endpoints (configured via centralized gateway)
- **OpenAI**: Native OpenAI API
- **Anthropic**: Requires `anthropic` SDK (add to requirements if used directly)
- **Google Gemini**: Requires `google-generativeai` SDK (add when needed)
- **Other providers**: Many expose OpenAI-compatible endpoints; register them in `config/models/roles`.

### Step-by-Step Guide to Add a New API Provider

#### 1. Install Required SDK

Add the required Python package to `requirements.txt`:

```txt
# For OpenAI (if not already installed)
openai>=1.0.0

# For Anthropic Claude
anthropic>=0.18.0

# For Google Gemini
google-generativeai>=0.3.0

# For other OpenAI-compatible providers
# (usually just need openai package)
```

#### 2. Create API Client Function

In `app.py`, add a new function to create the API client. Here are examples for different providers:

##### Option A: OpenAI (OpenAI-compatible, easiest)

```python
def get_openai_client():
    """Get OpenAI API client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = _get_api_key_from_secrets("OPENAI_API_KEY")
    
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY is not set.")
        st.stop()
        return None
    
    return OpenAI(
        api_key=api_key
        # No base_url needed for OpenAI
    )
```

##### Option B: Anthropic Claude (Different SDK)

```python
import anthropic

def get_anthropic_client():
    """Get Anthropic Claude API client."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = _get_api_key_from_secrets("ANTHROPIC_API_KEY")
    
    if not api_key:
        st.error("⚠️ ANTHROPIC_API_KEY is not set.")
        st.stop()
        return None
    
    return anthropic.Anthropic(api_key=api_key)
```

##### Option C: Google Gemini (Different SDK)

```python
import google.generativeai as genai

def get_gemini_client():
    """Get Google Gemini API client."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = _get_api_key_from_secrets("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("⚠️ GOOGLE_API_KEY is not set.")
        st.stop()
        return None
    
    genai.configure(api_key=api_key)
    return genai
```

##### Option D: Generic OpenAI-Compatible Provider

```python
def get_custom_client():
    """Get custom OpenAI-compatible API client."""
    api_key = os.getenv("CUSTOM_API_KEY")
    base_url = os.getenv("CUSTOM_API_BASE_URL", "https://api.custom-provider.com/v1")
    
    if not api_key:
        api_key = _get_api_key_from_secrets("CUSTOM_API_KEY")
    
    if not api_key:
        st.error("⚠️ CUSTOM_API_KEY is not set.")
        st.stop()
        return None
    
    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )
```

#### 3. Update API Call Functions

You need to modify three functions in `app.py`:

##### A. Main Chat Function (around line 850)

**Current (OpenAI-compatible default):**
```python
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=api_messages,
    temperature=0.7,
    max_tokens=2000
)
ai_response = response.choices[0].message.content
```

**For Anthropic Claude:**
```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2000,
    temperature=0.7,
    messages=api_messages
)
ai_response = response.content[0].text
```

**For Google Gemini:**
```python
model = client.GenerativeModel('gemini-pro')
response = model.generate_content(
    "\n".join([f"{msg['role']}: {msg['content']}" for msg in api_messages]),
    generation_config={
        'temperature': 0.7,
        'max_output_tokens': 2000,
    }
)
ai_response = response.text
```

##### B. SRS Generation Function (around line 427)

Update `generate_ieee830_srs_from_conversation()` to use your provider's API format.

**For OpenAI-compatible (current):**
```python
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[...],
    temperature=0.3,
    max_tokens=4000
)
srs_content = response.choices[0].message.content
```

**For Anthropic:**
```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4000,
    temperature=0.3,
    messages=[...]
)
srs_content = response.content[0].text
```

##### C. Summarization Function (in `memory.py`, around line 415)

Update `summarize_old_messages()` method in `memory.py`:

**For OpenAI-compatible (current):**
```python
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[...],
    temperature=0.3,
    max_tokens=500
)
summary = response.choices[0].message.content
```

**For Anthropic:**
```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=500,
    temperature=0.3,
    messages=[...]
)
summary = response.content[0].text
```

#### 4. Create a Unified API Wrapper (Recommended)

For cleaner code, create a wrapper function that abstracts API differences:

```python
def call_api(client, provider, model, messages, temperature=0.7, max_tokens=2000):
    """
    Unified API call wrapper that handles different providers.
    
    Args:
        client: API client instance
        provider: "deepseek", "openai", "anthropic", "gemini"
        model: Model name
        messages: List of message dictionaries
        temperature: Temperature setting
        max_tokens: Maximum tokens to generate
    
    Returns:
        str: Generated text response
    """
    if provider in ["deepseek", "openai"]:
        # OpenAI-compatible API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    elif provider == "anthropic":
        # Anthropic Claude API
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages
        )
        return response.content[0].text
    
    elif provider == "gemini":
        # Google Gemini API
        model_instance = client.GenerativeModel(model)
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        response = model_instance.generate_content(
            prompt,
            generation_config={
                'temperature': temperature,
                'max_output_tokens': max_tokens,
            }
        )
        return response.text
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")
```

Then use it throughout the code:

```python
# In main chat
client = get_openai_client()
ai_response = call_api(
    client=client,
    provider="openai",
    model="gpt-4o",
    messages=api_messages,
    temperature=0.7,
    max_tokens=2000
)
```

#### 5. Update Environment Variables

Add your API key to environment variables or `.streamlit/secrets.toml`:

**Environment Variable:**
```bash
export OPENAI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
export GOOGLE_API_KEY="your_key_here"
```

**Streamlit Secrets (`.streamlit/secrets.toml`):**
```toml
OPENAI_API_KEY = "your_key_here"
ANTHROPIC_API_KEY = "your_key_here"
GOOGLE_API_KEY = "your_key_here"
```

#### 6. Update Helper Functions

The helper `_get_api_key_from_secrets()` currently defaults to `CENTRALIZED_LLM_API_KEY`. To support per-provider secrets, you can either:

**Option A: Modify the function to accept a parameter (recommended):**

```python
def _get_api_key_from_secrets(key_name="CENTRALIZED_LLM_API_KEY"):
    """Safely get API key from Streamlit secrets."""
    try:
        api_key = st.secrets.get(key_name, None)
        return api_key if api_key else None
    except Exception:
        return None
```

Then use it in your client functions:
```python
api_key = _get_api_key_from_secrets("OPENAI_API_KEY")
api_key = _get_api_key_from_secrets("ANTHROPIC_API_KEY")
```

**Option B: Create provider-specific functions:**

```python
def _get_openai_api_key_from_secrets():
    """Get OpenAI API key from Streamlit secrets."""
    try:
        return st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        return None

def _get_anthropic_api_key_from_secrets():
    """Get Anthropic API key from Streamlit secrets."""
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", None)
    except Exception:
        return None
```

### Example: Adding OpenAI Support

Here's a complete example of adding OpenAI support:

1. **Update `requirements.txt`:**
   ```txt
   openai>=1.0.0
   ```

2. **Add to `app.py`:**
   ```python
   def get_openai_client():
       """Get OpenAI API client."""
       api_key = os.getenv("OPENAI_API_KEY")
       if not api_key:
           # Modify _get_api_key_from_secrets() to accept key_name parameter
           # or create a new function like _get_api_key_from_secrets_generic(key_name)
           api_key = _get_api_key_from_secrets_generic("OPENAI_API_KEY")
       
       if not api_key:
           st.error("⚠️ OPENAI_API_KEY is not set.")
           st.info("Please set OPENAI_API_KEY as an environment variable or in .streamlit/secrets.toml")
           st.stop()
           return None
       
       return OpenAI(api_key=api_key)
   ```
   
   **Note:** You'll need to create a generic version of `_get_api_key_from_secrets()` that accepts a key name parameter, or create provider-specific functions as shown in Step 6.

3. **Update main chat function:**
   ```python
   # Replace get_deepseek_client() with get_openai_client()
   client = get_openai_client()
   
   # Change model name
   response = client.chat.completions.create(
       model="gpt-4o",  # or "gpt-4o-mini", "gpt-4-turbo", etc.
       messages=api_messages,
       temperature=0.7,
       max_tokens=2000
   )
   ```

4. **Update SRS generation:**
   ```python
   response = client.chat.completions.create(
       model="gpt-4o",
       messages=[...],
       temperature=0.3,
       max_tokens=4000
   )
   ```

5. **Set environment variable:**
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

### Best Practices

1. **Use Environment Variables**: Always prefer environment variables over hardcoded keys
2. **Error Handling**: Wrap API calls in try-except blocks
3. **Rate Limiting**: Implement rate limiting for production use
4. **Token Management**: Monitor token usage, especially for paid APIs
5. **Model Selection**: Allow users to select models via UI (optional)
6. **Fallback**: Consider implementing fallback to another provider if one fails
7. **Logging**: Log API calls for debugging and monitoring

### Testing Your Implementation

1. **Test API Connection:**
   ```python
   client = get_your_client()
   if client:
       st.success("✅ API client created successfully")
   ```

2. **Test API Call:**
   ```python
   try:
       response = call_api(client, provider, model, test_messages)
       st.success(f"✅ API call successful: {response[:50]}...")
   except Exception as e:
       st.error(f"❌ API call failed: {str(e)}")
   ```

3. **Test All Functions:**
   - Main chat conversation
   - SRS generation
   - Context summarization

### Common Issues and Solutions

1. **Import Errors**: Make sure the SDK is installed (`pip install -r requirements.txt`)
2. **API Key Not Found**: Check environment variables and secrets.toml
3. **Rate Limits**: Implement retry logic with exponential backoff
4. **Token Limits**: Adjust `max_tokens` parameter based on model limits
5. **Model Not Found**: Verify model name is correct for your provider
6. **Timeout Errors**: Increase timeout settings in API client configuration

### Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic Claude API Documentation](https://docs.anthropic.com/)
- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [DeepSeek API Documentation](https://platform.deepseek.com/api-docs/)
- [OpenAI-Compatible APIs List](https://github.com/openai/openai-python#openai-compatible-apis)

### Contributing

If you add support for a new API provider, consider:
1. Adding it to this documentation
2. Creating a unified API wrapper function
3. Adding tests for the new provider
4. Updating `requirements.txt` with new dependencies
