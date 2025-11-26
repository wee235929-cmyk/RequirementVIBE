# Streamlit Cloud Deployment Instructions

## Prerequisites

1. A GitHub account
2. A Streamlit Cloud account (free at https://streamlit.io/cloud)
3. Your DeepSeek API key

## Deployment Steps

### 1. Push Your Code to GitHub

1. Initialize a git repository (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit - ReqVibe app"
   ```

2. Create a new repository on GitHub (e.g., `ReqVibe`)

3. Push your code to GitHub:
   ```bash
   git remote add origin https://github.com/yourusername/ReqVibe.git
   git branch -M main
   git push -u origin main
   ```

### 2. Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository and branch (usually `main`)
5. Set the main file path to: `app.py`
6. Click "Deploy"

### 3. Configure Secrets (API Key)

1. In your Streamlit Cloud app dashboard, click on "⋮" (three dots) → "Settings"
2. Go to "Secrets" tab
3. Add your DeepSeek API key:
   ```toml
   DEEPSEEK_API_KEY = "YOUR_API_KEY_HERE"
   ```
4. Click "Save"

### 4. Restart Your App

1. After saving secrets, go to "Manage app"
2. Click "⋮" → "Restart app"
3. Your app will restart with the new configuration

## File Structure for Deployment

```
RequirenebtVIBE/
├── app.py                  # Main application file
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit configuration
└── README.md              # Project documentation
```

## Important Notes

- **Never commit your API key to GitHub!** Always use Streamlit Cloud's Secrets feature.
- The app will automatically redeploy when you push changes to the main branch.
- Make sure `requirements.txt` includes all necessary dependencies.
- The `.streamlit/config.toml` file is optional but recommended for better app configuration.
- The `openai` package in requirements.txt is required because DeepSeek API uses an OpenAI-compatible SDK.

## Voice Transcription Model (Git LFS)

The Whisper **base** model (~150 MB) is stored in `models/whisper/base.pt` via Git LFS so Streamlit Cloud can start recording immediately.

### Download or refresh the model locally

```bash
python scripts/download_whisper_model.py    # defaults to base
```

### Commit the model with Git LFS

```bash
git lfs install                      # once per machine
git add models/whisper/base.pt .gitattributes
git commit -m "Add Whisper base model for Streamlit"
git push origin main
```

If you would rather download on first run, remove it from Git tracking:

```bash
git rm --cached models/whisper/base.pt
git commit -m "Let Whisper download model at runtime"
git push origin main
```

## Document Processing on Streamlit Cloud

### Automatic Cache Configuration

The app **automatically configures cache directories** for document processing tools (Docling, RapidOCR, Hugging Face models) when deployed to Streamlit Cloud. This ensures models can be downloaded to writable locations without permission issues.

**How it works**:
1. The app detects if it's running on Streamlit Cloud (checks for `/mount/src` directory)
2. Automatically finds your repository name
3. Creates `.cache/` directories in your project root
4. Sets environment variables (`RAPIDOCR_HOME`, `HF_HOME`, `HF_HUB_CACHE`) to point to these directories
5. All model downloads go to writable locations within your repo

**No manual configuration needed** - the app handles this automatically!

### Manual Override (Advanced)

If you need to override the automatic configuration, you can set these environment variables in Streamlit Cloud Secrets:

```toml
RAPIDOCR_HOME = "/mount/src/RequirementVIBE/.cache/rapidocr"
HF_HOME = "/mount/src/RequirementVIBE/.cache/huggingface"
HF_HUB_CACHE = "/mount/src/RequirementVIBE/.cache/huggingface/hub"
```

**Note**: Replace `RequirementVIBE` with your actual repository name.

### Troubleshooting Document Processing

**Issue**: Document processing fails with permission errors

**Solution**:
1. Check that `.cache/` directory is not in `.gitignore` (it should be ignored)
2. Verify the app logs show cache directories being configured
3. If using manual override, ensure paths match your repository name
4. Restart the app after changing environment variables

**Issue**: Models download slowly on first use

**Solution**: This is normal! Models are cached after first download:
- Docling layout model (~500MB) - downloaded once
- RapidOCR models (~3MB) - downloaded once
- SentenceTransformer embeddings (~80MB) - downloaded once

Subsequent document processing will be faster.

## Troubleshooting

### Repository fails to clone

1. Double-check the repository and branch names in Streamlit Cloud (case-sensitive)
2. Ensure Git LFS objects are uploaded:
   ```bash
   git lfs push origin main --all
   git push origin main
   ```
3. Re-authorize Streamlit Cloud to access your GitHub account if prompted

### API key not working
- Verify the secret is set correctly in Streamlit Cloud (`DEEPSEEK_API_KEY`)
- Remove extra quotes or whitespace around the key
- Restart the app after updating secrets

### Voice model download hangs
- Confirm `models/whisper/base.pt` exists in GitHub (if using the preloaded model)
- Otherwise expect the first transcription to take a few minutes while Whisper downloads the base model on Streamlit Cloud

### App loads but shows error
- Check the app logs in Streamlit Cloud
- Ensure all dependencies are listed in `requirements.txt`
- Verify the DeepSeek API is reachable from Streamlit servers

## Environment Variables

### Required Variables

At minimum set:

| Variable | Where to set | Purpose |
|----------|--------------|---------|
| `DEEPSEEK_API_KEY` | Streamlit Cloud → Secrets | LLM access |
| `CENTRALIZED_LLM_API_KEY` | `.env` or Secrets | Gateway key (if different from DeepSeek) |
| `UNSTRUCTURED_API_KEY` | `.env` or Secrets | Document processing (only if using Unstructured API fallback) |

### Optional Variables

| Variable | Where to set | Purpose |
|----------|--------------|---------|
| `VOICE_TRANSCRIBE_*` | `.env` or Secrets | Override Whisper defaults |
| `RAPIDOCR_HOME` | `.env` or Secrets | Custom RapidOCR cache directory (auto-configured if not set) |
| `HF_HOME` | `.env` or Secrets | Custom Hugging Face cache directory (auto-configured if not set) |
| `HF_HUB_CACHE` | `.env` or Secrets | Custom Hugging Face Hub cache directory (auto-configured if not set) |

### Document Processing Cache Configuration

**Important for Streamlit Cloud**: The app automatically configures cache directories for document processing (Docling, RapidOCR, Hugging Face models). 

**Automatic Configuration**:
- The app detects if it's running on Streamlit Cloud (`/mount/src` exists)
- Automatically creates `.cache/` directories in the project root
- Sets `RAPIDOCR_HOME`, `HF_HOME`, and `HF_HUB_CACHE` to writable locations
- No manual configuration needed in most cases

**Manual Override** (if needed):
If you need to override the automatic configuration, set these in Streamlit Cloud Secrets:
```toml
RAPIDOCR_HOME = "/mount/src/RequirementVIBE/.cache/rapidocr"
HF_HOME = "/mount/src/RequirementVIBE/.cache/huggingface"
HF_HUB_CACHE = "/mount/src/RequirementVIBE/.cache/huggingface/hub"
```

**Note**: Replace `RequirementVIBE` with your actual repository name if different.

## Custom Domain (Optional)

Streamlit Cloud allows you to use a custom domain:
1. Go to app settings
2. Navigate to "Custom domain"
3. Follow the instructions to configure DNS

