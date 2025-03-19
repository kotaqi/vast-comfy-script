#!/bin/bash
# ComfyUI setup script for Vast.ai

cd /workspace/
# Cause the script to exit on failure
set -eo pipefail

# Activate the main virtual environment
. /venv/main/bin/activate

# Install required packages
pip install torch torchvision torchaudio xformers opencv-python-headless

# Clone ComfyUI repository
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install ComfyUI requirements
pip install -r requirements.txt

# Create a consolidated requirements file for deduplication
touch ../consolidated_requirements.txt

# Create custom_nodes directory
mkdir -p custom_nodes
cd custom_nodes

# Function to clone a repository and extract its requirements
clone_and_extract_requirements() {
    repo_url=$1
    folder_name=$2
    echo "Cloning $folder_name from $repo_url..."
    git clone "$repo_url" "$folder_name"
    
    # Check if requirements.txt exists and add to consolidated file
    if [ -f "$folder_name/requirements.txt" ]; then
        echo "# Requirements from $folder_name" >> ../../consolidated_requirements.txt
        cat "$folder_name/requirements.txt" >> ../../consolidated_requirements.txt
        echo "" >> ../../consolidated_requirements.txt
    fi
    
    # Check for install.py and run it if exists
    if [ -f "$folder_name/install.py" ]; then
        echo "Running install script for $folder_name..."
        python "$folder_name/install.py"
    fi
}

# Clone custom nodes (including a subset of the most important ones)
clone_and_extract_requirements "https://github.com/ltdrdata/ComfyUI-Manager.git" "ComfyUI-Manager"
clone_and_extract_requirements "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git" "ComfyUI-AnimateDiff-Evolved"
clone_and_extract_requirements "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git" "ComfyUI-Impact-Pack"
clone_and_extract_requirements "https://github.com/Fannovel16/comfyui-inpaint-nodes.git" "comfyui-inpaint-nodes"
clone_and_extract_requirements "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git" "ComfyUI-VideoHelperSuite"
clone_and_extract_requirements "https://github.com/Gourieff/comfyui-reactor-node.git" "comfyui-reactor-node"
clone_and_extract_requirements "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git" "ComfyUI_IPAdapter_plus"
clone_and_extract_requirements "https://github.com/cubiq/ComfyUI_essentials.git" "ComfyUI_essentials"
clone_and_extract_requirements "https://github.com/rgthree/rgthree-comfy.git" "rgthree-comfy"
clone_and_extract_requirements "https://github.com/WASasquatch/was-node-suite-comfyui.git" "was-node-suite-comfyui"

# Back to main directory
cd ../..

# Process consolidated requirements to remove duplicates and conflicts
python -c "
import re
seen = set()
with open('consolidated_requirements.txt', 'r') as f, open('filtered_requirements.txt', 'w') as out:
    for line in f:
        if line.startswith('#') or not line.strip():
            out.write(line)
            continue
        
        # Extract package name (ignoring version constraints)
        match = re.match(r'^([a-zA-Z0-9_\-]+)', line.strip())
        if match:
            package = match.group(1).lower()
            if package not in seen:
                seen.add(package)
                out.write(line)
"

# Install filtered requirements
pip install -r filtered_requirements.txt

# Return to ComfyUI directory
cd ComfyUI

# Create models directory structure
mkdir -p models/{animatediff_models,checkpoints,clip,clip_vision,controlnet,diffusion_models,embeddings,ipadapter,loras,unet,upscale_models,vae}

# Function for downloading models with retry logic
download_with_retry() {
    url=$1
    target_dir=$2
    filename=$3
    max_retries=5
    retry_delay=10
    retries=0

    mkdir -p "$target_dir"
    
    echo "Downloading $filename to $target_dir..."
    
    until [ $retries -ge $max_retries ]
    do
        wget -c -q --show-progress --connect-timeout 30 "$url" -O "$target_dir/$filename" && break
        
        retries=$((retries+1))
        echo "Download failed. Retrying in $retry_delay seconds... ($retries/$max_retries)"
        sleep $retry_delay
    done

    if [ $retries -ge $max_retries ]; then
        echo "Download failed after $max_retries attempts."
        return 1
    else
        echo "Download completed for $filename."
        return 0
    fi
}

# Download essential models (a subset to get started)
download_with_retry "https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/resolve/main/flux1-dev-bnb-nf4.safetensors" "models/checkpoints" "flux1-dev-bnb-nf4.safetensors"
download_with_retry "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors" "models/checkpoints" "svd_xt.safetensors"
download_with_retry "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors" "models/vae" "vae-ft-mse-840000-ema-pruned.safetensors"
download_with_retry "https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/diffusion_pytorch_model.safetensors" "models/controlnet" "control_v11p_sd15_openpose.safetensors"
download_with_retry "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors" "models/ipadapter" "ip-adapter_sd15.safetensors"
download_with_retry "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt" "models/animatediff_models" "mm_sd_v15_v2.ckpt"

# Set up supervisor configuration for ComfyUI
cat > /etc/supervisor/conf.d/comfyui.conf << EOF
[program:comfyui]
command=/venv/main/bin/python /workspace/ComfyUI/main.py --listen 0.0.0.0 --port 8188
directory=/workspace/ComfyUI
autostart=true
autorestart=true
stdout_logfile=/var/log/comfyui.log
redirect_stderr=true
EOF

# Create a wrapper script for supervisor
cat > /opt/supervisor-scripts/comfyui.sh << EOF
#!/bin/bash
cd /workspace/ComfyUI
exec /venv/main/bin/python main.py --listen 0.0.0.0 --port 8188
EOF
chmod +x /opt/supervisor-scripts/comfyui.sh

# Configure the instance portal
rm -f /etc/portal.yaml
export PORTAL_CONFIG="localhost:8188:18188:/comfyui/:ComfyUI"

# Reload Supervisor to apply changes
supervisorctl reload

echo "ComfyUI setup complete! Access it through the 'ComfyUI' button in your Vast.ai instance portal."