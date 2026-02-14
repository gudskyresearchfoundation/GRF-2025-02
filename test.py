#!/usr/bin/env python3
"""
Lightning.ai Diagnostic Script for Balaramaji
Checks system, Ollama, and identifies issues
Saves output to text.txt
"""

import sys
import os
import subprocess
import requests
import json
from pathlib import Path
from datetime import datetime

# Create output file
output_file = open('text.txt', 'w')

def log_print(msg):
    """Print to console and write to file"""
    print(msg)
    output_file.write(msg + '\n')
    output_file.flush()

log_print("=" * 70)
log_print("🔍 LIGHTNING.AI BALARAMAJI DIAGNOSTIC")
log_print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_print("=" * 70)
log_print("")

# Color codes for terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_success(msg):
    colored_msg = f"{Colors.GREEN}✅ {msg}{Colors.END}"
    plain_msg = f"✅ {msg}"
    print(colored_msg)
    output_file.write(plain_msg + '\n')
    output_file.flush()

def print_error(msg):
    colored_msg = f"{Colors.RED}❌ {msg}{Colors.END}"
    plain_msg = f"❌ {msg}"
    print(colored_msg)
    output_file.write(plain_msg + '\n')
    output_file.flush()

def print_warning(msg):
    colored_msg = f"{Colors.YELLOW}⚠️  {msg}{Colors.END}"
    plain_msg = f"⚠️  {msg}"
    print(colored_msg)
    output_file.write(plain_msg + '\n')
    output_file.flush()

def print_info(msg):
    colored_msg = f"{Colors.BLUE}ℹ️  {msg}{Colors.END}"
    plain_msg = f"ℹ️  {msg}"
    print(colored_msg)
    output_file.write(plain_msg + '\n')
    output_file.flush()

# ============================================================================
# 1. SYSTEM CHECKS
# ============================================================================
log_print("1️⃣  SYSTEM INFORMATION")
log_print("-" * 70)

# Python version
python_version = sys.version.split()[0]
log_print(f"Python Version: {python_version}")
if sys.version_info >= (3, 8):
    print_success("Python version is compatible")
else:
    print_error("Python 3.8+ required")

# RAM
try:
    with open('/proc/meminfo', 'r') as f:
        meminfo = f.read()
        total_ram = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1]) // 1024
        free_ram = int([line for line in meminfo.split('\n') if 'MemAvailable' in line][0].split()[1]) // 1024
    
    log_print(f"Total RAM: {total_ram} MB ({total_ram/1024:.1f} GB)")
    log_print(f"Available RAM: {free_ram} MB ({free_ram/1024:.1f} GB)")
    
    if total_ram < 24000:
        print_warning(f"Low RAM: {total_ram/1024:.1f}GB (24GB+ recommended for Qwen 32B)")
        print_info("RECOMMENDATION: Use qwen2.5:14b instead (needs 16GB)")
    else:
        print_success(f"Sufficient RAM: {total_ram/1024:.1f}GB")
except Exception as e:
    print_error(f"Cannot read RAM info: {e}")

# Disk space
try:
    disk = subprocess.check_output(['df', '-h', '/']).decode().split('\n')[1].split()
    log_print(f"Disk Space Available: {disk[3]}")
    
    disk_gb = float(disk[3].replace('G', '').replace('T', '000'))
    if disk_gb < 25:
        print_error(f"Low disk space: {disk[3]} (need 25GB+ for Qwen 32B)")
    else:
        print_success(f"Sufficient disk space: {disk[3]}")
except Exception as e:
    print_error(f"Cannot read disk info: {e}")

# GPU
log_print("\nGPU Information:")
try:
    nvidia_smi = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader']).decode().strip()
    log_print(f"GPU: {nvidia_smi}")
    print_success("NVIDIA GPU detected")
except:
    print_warning("No NVIDIA GPU detected (will use CPU - slower)")

log_print("")

# ============================================================================
# 2. OLLAMA CHECKS
# ============================================================================
log_print("2️⃣  OLLAMA STATUS")
log_print("-" * 70)

# Check if Ollama is installed
try:
    ollama_version = subprocess.check_output(['ollama', '--version'], stderr=subprocess.STDOUT).decode().strip()
    log_print(f"Ollama Version: {ollama_version}")
    print_success("Ollama is installed")
except FileNotFoundError:
    print_error("Ollama is NOT installed")
    print_info("Install: curl -fsSL https://ollama.com/install.sh | sh")
    output_file.close()
    sys.exit(1)
except Exception as e:
    print_error(f"Ollama check failed: {e}")

# Check if Ollama is running
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
log_print(f"\nChecking Ollama at: {ollama_host}")

try:
    response = requests.get(f"{ollama_host}/api/tags", timeout=5)
    if response.status_code == 200:
        print_success("Ollama service is running")
        
        # List models
        models = response.json().get("models", [])
        log_print(f"\nInstalled Models ({len(models)}):")
        
        qwen_found = False
        qwen_32b = False
        qwen_14b = False
        qwen_7b = False
        
        for model in models:
            name = model.get("name")
            size_gb = model.get("size", 0) / (1024**3)
            log_print(f"  - {name} ({size_gb:.1f} GB)")
            
            if "qwen2.5:32b" in name:
                qwen_32b = True
                qwen_found = True
                print_success(f"    Qwen 32B found!")
            elif "qwen2.5:14b" in name:
                qwen_14b = True
                qwen_found = True
                print_success(f"    Qwen 14B found!")
            elif "qwen2.5:7b" in name:
                qwen_7b = True
                qwen_found = True
                print_success(f"    Qwen 7B found!")
        
        if not qwen_found:
            print_error("No Qwen 2.5 model installed")
            print_info("Install: ollama pull qwen2.5:32b")
            print_info("Or for less RAM: ollama pull qwen2.5:14b")
        elif not qwen_32b:
            print_warning("Qwen 32B not installed (using smaller model)")
            if qwen_14b:
                print_info("Using Qwen 14B - update balaramaji.py: self.model = 'qwen2.5:14b'")
            elif qwen_7b:
                print_info("Using Qwen 7B - update balaramaji.py: self.model = 'qwen2.5:7b'")
        
    else:
        print_error(f"Ollama responded with status {response.status_code}")
        
except requests.exceptions.ConnectionError:
    print_error("Cannot connect to Ollama service")
    print_info("Start Ollama: ollama serve")
    print_info("Or run in background: nohup ollama serve > /tmp/ollama.log 2>&1 &")
    print_info("For Lightning.ai: OLLAMA_HOST=0.0.0.0:11434 ollama serve &")
except Exception as e:
    print_error(f"Ollama connection failed: {e}")

log_print("")

# ============================================================================
# 3. PROJECT STRUCTURE
# ============================================================================
log_print("3️⃣  PROJECT STRUCTURE")
log_print("-" * 70)

# Check if we're in the right directory
cwd = Path.cwd()
log_print(f"Current Directory: {cwd}")

# Expected files
expected_files = {
    "src/models/balaramaji.py": "Balaramaji model file",
    "src/backend.py": "Backend API",
    "frontend/src/js/app.js": "Frontend JavaScript",
    "src/models/__init__.py": "Models package init"
}

log_print("\nChecking required files:")
for file_path, description in expected_files.items():
    full_path = cwd / file_path
    if full_path.exists():
        print_success(f"{file_path} - {description}")
    else:
        print_error(f"{file_path} - MISSING!")

log_print("")

# ============================================================================
# 4. PYTHON DEPENDENCIES
# ============================================================================
log_print("4️⃣  PYTHON DEPENDENCIES")
log_print("-" * 70)

required_packages = {
    "fastapi": "Web framework",
    "uvicorn": "ASGI server",
    "requests": "HTTP library",
    "pydantic": "Data validation"
}

log_print("Checking Python packages:")
for package, description in required_packages.items():
    try:
        __import__(package)
        print_success(f"{package} - {description}")
    except ImportError:
        print_error(f"{package} - NOT INSTALLED")
        print_info(f"Install: pip install {package}")

log_print("")

# ============================================================================
# 5. TEST BALARAMAJI IMPORT
# ============================================================================
log_print("5️⃣  BALARAMAJI IMPORT TEST")
log_print("-" * 70)

try:
    # Try to import
    sys.path.insert(0, str(cwd))
    from src.models.balaramaji import BalaramajiAssistant, get_balaramaji_assistant
    print_success("Balaramaji module imports successfully")
    
    # Try to initialize
    try:
        log_print("\nAttempting to initialize Balaramaji...")
        assistant = BalaramajiAssistant(ollama_host=ollama_host)
        print_success("Balaramaji initialized successfully")
        
        # Try a test message
        log_print("\nTesting basic response generation...")
        try:
            response = assistant.generate_response(
                user_message="Hello, test message",
                analysis_context=None,
                chat_history=None
            )
            
            if response.get('status') == 'success':
                print_success("Response generated successfully")
                log_print(f"  Source: {response.get('source')}")
                log_print(f"  Model: {response.get('model')}")
                log_print(f"  Response preview: {response.get('response')[:100]}...")
            else:
                print_error(f"Response failed: {response.get('error')}")
                
        except Exception as e:
            print_error(f"Response generation failed: {e}")
            log_print(f"Error type: {type(e).__name__}")
            import traceback
            error_trace = traceback.format_exc()
            log_print(error_trace)
            
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        log_print(f"Error type: {type(e).__name__}")
        import traceback
        error_trace = traceback.format_exc()
        log_print(error_trace)
        
except ImportError as e:
    print_error(f"Cannot import Balaramaji: {e}")
    print_info("Check that balaramaji.py exists in src/models/")
except Exception as e:
    print_error(f"Import test failed: {e}")
    import traceback
    error_trace = traceback.format_exc()
    log_print(error_trace)

log_print("")

# ============================================================================
# 6. RECOMMENDATIONS
# ============================================================================
log_print("6️⃣  RECOMMENDATIONS")
log_print("-" * 70)

log_print("\n📋 Based on the diagnostics above, here's what to do:\n")

log_print("If Ollama is NOT installed:")
log_print("  1. curl -fsSL https://ollama.com/install.sh | sh")
log_print("  2. ollama serve")
log_print("  3. ollama pull qwen2.5:32b")
log_print("")

log_print("If Ollama is NOT running:")
log_print("  1. ollama serve")
log_print("  OR for background: nohup ollama serve > /tmp/ollama.log 2>&1 &")
log_print("")

log_print("If Qwen 32B is NOT installed:")
log_print("  1. ollama pull qwen2.5:32b")
log_print("  2. Wait 15-30 minutes for download (~20GB)")
log_print("")

log_print("If you have low RAM (<24GB):")
log_print("  1. Use smaller model: ollama pull qwen2.5:14b")
log_print("  2. Update balaramaji.py line 27: self.model = 'qwen2.5:14b'")
log_print("")

log_print("If imports fail:")
log_print("  1. pip install -r requirements.txt")
log_print("  2. Check file paths are correct")
log_print("")

log_print("Common Lightning.ai specific issues:")
log_print("  1. Ollama may need to bind to 0.0.0.0 instead of localhost")
log_print("  2. Run: OLLAMA_HOST=0.0.0.0:11434 ollama serve &")
log_print("  3. Set environment: export OLLAMA_HOST=http://0.0.0.0:11434")
log_print("")

log_print("=" * 70)
log_print("🏁 DIAGNOSTIC COMPLETE")
log_print("=" * 70)
log_print("")
log_print("📝 Results saved to text.txt")
log_print("📤 Share text.txt for specific help")

# Close output file
output_file.close()

print(f"\n✅ Diagnostic complete! Results saved to: {os.path.abspath('text.txt')}")