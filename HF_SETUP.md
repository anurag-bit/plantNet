# HuggingFace Setup for PlantNet Automated Upload

## Quick Setup Commands

### 🔧 Method 1: Environment Variables (Recommended)

**Linux/macOS - Current Session:**
```bash
export HUGGINGFACE_TOKEN='your_token_here'
export HUGGINGFACE_USERNAME='your_username_here'  # optional
```

**Linux/macOS - Permanent Setup:**
```bash
echo 'export HUGGINGFACE_TOKEN="your_token_here"' >> ~/.bashrc
echo 'export HUGGINGFACE_USERNAME="your_username_here"' >> ~/.bashrc  # optional
source ~/.bashrc
```

**Windows Command Prompt:**
```cmd
set HUGGINGFACE_TOKEN=your_token_here
set HUGGINGFACE_USERNAME=your_username_here
```

**Windows PowerShell - Permanent:**
```powershell
[Environment]::SetEnvironmentVariable('HUGGINGFACE_TOKEN', 'your_token_here', 'User')
[Environment]::SetEnvironmentVariable('HUGGINGFACE_USERNAME', 'your_username_here', 'User')
```

### 🔧 Method 2: Interactive Setup Script

```bash
./setup_hf_credentials.sh
```

### 🔧 Method 3: Command Line Arguments

```bash
python automated_huggingface_upload.py --token your_token_here --username your_username_here
```

## 📝 Getting Your Token

1. Visit: https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "PlantNet Upload")
4. Select "Write" permissions
5. Copy the token

## ✅ Verify Setup

Test that your credentials are properly configured:

```bash
python -c "import os; print('✅ Token set!' if os.getenv('HUGGINGFACE_TOKEN') else '❌ Token not found')"
```

## 🚀 Run the Upload

Once your credentials are set up:

```bash
python automated_huggingface_upload.py
```

Or with custom options:
```bash
python automated_huggingface_upload.py --repo-name my-custom-plant-model --private
```

## 🔒 Security Notes

- Never commit tokens to version control
- Use environment variables instead of hardcoding
- Keep your tokens secure and rotate them regularly
- Use the minimum required permissions (write access for uploads)