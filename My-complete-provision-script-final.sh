#!/bin/bash

# Environment setup
source /venv/main/bin/activate
COMFYUI_DIR=${WORKSPACE}/ComfyUI

# Token placeholders - replace these with your actual tokens
GITHUB_TOKEN="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"
HF_TOKEN="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"
CIVITAI_TOKEN="17f66905819af71b6933839889bfd3c9"

# Export tokens for use in commands
export GITHUB_TOKEN
export HF_TOKEN
export CIVITAI_TOKEN

### FUNCTIONS ###

provisioning_get_apt_packages() {
    if [ ${#APT_PACKAGES[@]} -gt 0 ]; then
        sudo apt-get update || { echo "apt-get update failed"; exit 1; }
        sudo apt-get install -y "${APT_PACKAGES[@]}" || { echo "apt-get install failed"; exit 1; }
    fi
}

provisioning_get_pip_packages() {
    if [ ${#PIP_PACKAGES[@]} -gt 0 ]; then
        pip install --no-cache-dir "${PIP_PACKAGES[@]}" || { echo "pip install failed"; exit 1; }
    fi
}

provisioning_get_nodes() {
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="${COMFYUI_DIR}/custom_nodes/${dir}"
        requirements="${path}/requirements.txt"
        if [ -d "$path" ]; then
            if [ "${AUTO_UPDATE,,}" != "false" ]; then
                printf "Updating node: %s...\n" "${repo}"
                ( cd "$path" && git pull ) || echo "Failed to update $repo"
                [ -e "$requirements" ] && pip install --no-cache-dir -r "$requirements" || echo "Failed to install requirements for $repo"
            fi
        else
            printf "Downloading node: %s...\n" "${repo}"
            if curl -s -I "$repo" | grep -q "HTTP/2 200"; then
                if [ -n "$GITHUB_TOKEN" ]; then
                    git clone "https://$GITHUB_TOKEN@github.com/${repo#https://github.com/}" "$path" --recursive || echo "Failed to clone $repo with token"
                else
                    git clone "$repo" "$path" --recursive || echo "Failed to clone $repo"
                fi
                [ -e "$requirements" ] && pip install --no-cache-dir -r "$requirements" || echo "Failed to install requirements for $repo"
            else
                echo "Skipping $repo: Repository not found or inaccessible (HTTP status not 200)."
            fi
        fi
    done
}

provisioning_get_files() {
    [ -z "$2" ] && return 1
    base_dir="$1"
    shift
    arr=("$@")
    printf "Processing %s model(s) to %s...\n" "${#arr[@]}" "$base_dir"
    for entry in "${arr[@]}"; do
        url=$(echo "$entry" | rev | cut -d':' -f3- | rev)
        subdir=$(echo "$entry" | rev | cut -d':' -f2 | rev)
        filename=$(echo "$entry" | rev | cut -d':' -f1 | rev)
        if [ -z "$subdir" ] || [ -z "$filename" ]; then
            target_dir="$base_dir"
            target_file=$(basename "$url")
        else
            target_dir="${COMFYUI_DIR%/}/models/${subdir}"
            target_file="$filename"
        fi
        mkdir -p "$target_dir"
        printf "Downloading: %s to %s/%s\n" "$url" "$target_dir" "$target_file"
        provisioning_download "$url" "$target_dir" "$target_file" || echo "Failed to download $url"
        printf "\n"
    done
}

provisioning_print_header() {
    printf "\n##############################################\n#                                            #\n#          Provisioning container            #\n#                                            #\n#         This will take some time           #\n#                                            #\n# Your container will be ready on completion #\n#                                            #\n##############################################\n\n"
}

provisioning_print_end() {
    printf "\nProvisioning complete: Application will start now\n\n"
}

provisioning_has_valid_hf_token() {
    [ -z "$HF_TOKEN" ] && return 1
    url="https://huggingface.co/api/whoami-v2"
    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $HF_TOKEN" \
        -H "Content-Type: application/json")
    [ "$response" -eq 200 ]
}

provisioning_has_valid_civitai_token() {
    [ -z "$CIVITAI_TOKEN" ] && return 1
    url="https://civitai.com/api/v1/models?hidden=1&limit=1"
    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $CIVITAI_TOKEN" \
        -H "Content-Type: application/json")
    [ "$response" -eq 200 ]
}

provisioning_download() {
    local url="$1"
    local dir="$2"
    local filename="$3"
    local auth_token=""
    if [ -n "$HF_TOKEN" ] && [[ "$url" =~ ^https://([a-zA-Z0-9_-]+\.)?huggingface\.co(/|$|\?) ]]; then
        auth_token="$HF_TOKEN"
    elif [ -n "$CIVITAI_TOKEN" ] && [[ "$url" =~ ^https://([a-zA-Z0-9_-]+\.)?civitai\.com(/|$|\?) ]]; then
        auth_token="$CIVITAI_TOKEN"
    fi
    if [ -n "$auth_token" ]; then
        wget --header="Authorization: Bearer $auth_token" -qnc --show-progress -e dotbytes="4M" -O "$dir/$filename" "$url" || { echo "Download failed: $url"; return 1; }
    else
        wget -qnc --show-progress -e dotbytes="4M" -O "$dir/$filename" "$url" || { echo "Download failed: $url"; return 1; }
    fi
}

create_extra_model_paths() {
    local yaml_file="${COMFYUI_DIR}/extra_model_paths.yaml"
    echo "Creating extra_model_paths.yaml to use checkpoints2 folder..."
    cat > "$yaml_file" << EOF
models:
  base_path: ${COMFYUI_DIR%/}/
  is_default: true
  checkpoints: models/checkpoints2/
EOF
    echo "extra_model_paths.yaml created successfully at ${yaml_file}"
}

provisioning_start() {
    provisioning_print_header
    provisioning_get_apt_packages
    provisioning_get_pip_packages
# Install ComfyUI if not present
    if [ ! -d "$COMFYUI_DIR" ]; then
        printf "Installing ComfyUI...\n"
        if [ -n "$GITHUB_TOKEN" ]; then
            git clone "https://$GITHUB_TOKEN@github.com/comfyanonymous/ComfyUI" "$COMFYUI_DIR" || { echo "Failed to clone ComfyUI"; exit 1; }
        else
            git clone "https://github.com/comfyanonymous/ComfyUI" "$COMFYUI_DIR" || { echo "Failed to clone ComfyUI"; exit 1; }
        fi
        # Install ComfyUI requirements
        pip install --no-cache-dir -r "$COMFYUI_DIR/requirements.txt" || { echo "Failed to install ComfyUI requirements"; exit 1; }
    fi
    
    provisioning_get_nodes
    workflows_dir="${COMFYUI_DIR}/user/default/workflows"
    mkdir -p "${workflows_dir}"
    provisioning_get_files "${workflows_dir}" "${WORKFLOWS[@]}"
    if provisioning_has_valid_hf_token; then
        UNET_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors:unet:flux1-dev.safetensors")
        VAE_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors:vae:ae.safetensors")
    else
        UNET_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors:unet:flux1-schnell.safetensors")
        VAE_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors:vae:ae.safetensors")
        sed -i 's/flux1-dev\.safetensors/flux1-schnell.safetensors/g' "${workflows_dir}/flux_dev_example.json" || echo "Failed to modify workflow file"
    fi
    provisioning_get_files "${COMFYUI_DIR}/models/clip" "${CLIP_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/checkpoints2" "${CHECKPOINT_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/vae" "${VAE_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/loras" "${LORA_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/animatediff_models" "${ANIMATEDIFF_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/unet" "${UNET_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/upscale_models" "${UPSCALE_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/ipadapter" "${IPADAPTER_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/pulid" "${PULID_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/instantid" "${INSTANTID_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/photomaker" "${PHOTOMAKER_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/inpaint" "${INPAINT_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/sams" "${SAM_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/llm_gguf" "${LLM_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/facerestore_models" "${FACERESTORE_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/vae_approx" "${VAE_APPROX_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/diffusion_models" "${DIFFUSION_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/controlnet" "${CONTROLNET_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/clip_vision" "${CLIP_VISION_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/style_models" "${STYLE_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/layer_model" "${LAYER_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/RMBG" "${RMBG_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/insightface" "${INSIGHTFACE_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/facedetection" "${FACEDETECTION_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/ultralytics" "${ULTRALYTICS_MODELS[@]}"
    if [ -f "${COMFYUI_DIR}/models/facedetection/shape_predictor_68_face_landmarks.dat.bz2" ]; then
        echo "Extracting shape_predictor_68_face_landmarks.dat.bz2..."
        bunzip2 -k "${COMFYUI_DIR}/models/facedetection/shape_predictor_68_face_landmarks.dat.bz2" || echo "Failed to extract .bz2 file"
    fi
    create_extra_model_paths
    provisioning_print_end
}

# Packages
APT_PACKAGES=(
    "git"
    "python3-pip"
    "python3-venv"
    "ffmpeg"
    "libgl1-mesa-glx"
    "libglib2.0-0"
)

PIP_PACKAGES=(
    "opencv-python"
    "torch"
    "torchvision"
    "torchaudio"
    "transformers"
    "diffusers"
    "accelerate"
    "xformers"
    "numpy"
    "pillow"
    "scipy"
)

# Nodes and Workflows
NODES=(
    "https://github.com/chrisgoringe/cg-use-everywhere"
    "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet"
    "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved"
    "https://github.com/crystian/ComfyUI-Crystools"
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"
    "https://github.com/griptape-ai/comfyui-griptape"
    "https://github.com/ltdrdata/ComfyUI-Impact-Pack"
    "https://github.com/ltdrdata/ComfyUI-impact-subpack"
    "https://github.com/ltdrdata/ComfyUI-Inspire-Pack"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/ltdrdata/ComfyUI-Manager"
    "https://github.com/huanngzh/ComfyUI-MVAdapter"
    "https://github.com/ZHO-ZHO-ZHO/ComfyUI-PhotoMaker-ZHO"
    "https://github.com/florestefano1975/ComfyUI-portrait-master"
    "https://github.com/Acly/ComfyUI-tooling-nodes"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/Fannovel16/ComfyUI_controlnet_aux"
    "https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet"
    "https://github.com/cubiq/ComfyUI_essentials"
    "https://github.com/comfyanonymous/ComfyUI_experiments"
    "https://github.com/FizzleDorf/ComfyUI_FizzNodes"
    "https://github.com/cubiq/ComfyUI_IPAdapter_plus"
    "https://github.com/SeargeDP/ComfyUI_Searge_LLM"
    "https://github.com/storyicon/ComfyUI_segment_anything"
    "https://github.com/TinyTerra/ComfyUI_tinyterraNodes"
    "https://github.com/ssitu/ComfyUI_UltimateSDUpscale"
    "https://github.com/rgthree/rgthree-comfy"
    "https://github.com/mcmonkeyprojects/sd-dynamic-thresholding"
    "https://github.com/BlenderNeko/ComfyUI_TiledKSampler"
    "https://github.com/WASasquatch/was-node-suite-comfyui"
    "https://github.com/kijai/ComfyUI-CogVideoXWrapper"
    "https://github.com/yolain/ComfyUI-Easy-Use"
    "https://github.com/kijai/ComfyUI-Florence2"
    "https://github.com/city96/ComfyUI-GGUF"
    "https://github.com/sipie800/ComfyUI-PuLID-Flux-Enhanced"
    "https://github.com/kijai/ComfyUI-SUPIR"
    "https://github.com/shiimizu/ComfyUI_smZNodes"
    "https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID"
    "https://github.com/kijai/ComfyUI-DepthAnythingV2"
    "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes"
    "https://github.com/farizrifqi/ComfyUI-Image-Saver"
    "https://github.com/kijai/ComfyUI-LivePortraitKJ"
    "https://github.com/sipherxyz/comfyui-art-venture"
    "https://github.com/Extraltodeus/ComfyUI-AutomaticCFG"
    "https://github.com/vienteck/ComfyUI-Chat-GPT-Integration"
    "https://github.com/theUpsider/ComfyUI-Logic"
    "https://github.com/PCMonsterx/ComfyUI-CSV-Loader"
    "https://github.com/Limitex/ComfyUI-Diffusers"
    "https://github.com/niknah/ComfyUI-F5-TTS"
    "https://github.com/pzc163/Comfyui-HunyuanDiT"
    "https://github.com/kijai/ComfyUI-IC-Light"
    "https://github.com/spacepxl/ComfyUI-Image-Filters"
    "https://github.com/Acly/comfyui-inpaint-nodes"
    "https://github.com/huchenlei/ComfyUI-layerdiffuse"
    "https://github.com/Fannovel16/ComfyUI-Marigold"
    "https://github.com/shadowcz007/comfyui-mixlab-nodes"
    "https://github.com/stavsap/comfyui-ollama"
    "https://github.com/kijai/ComfyUI-PyramidFlowWrapper"
    "https://github.com/omar92/ComfyUI-QualityOfLifeSuit_Omar92"
    "https://github.com/jiaxiangc/ComfyUI-ResAdapter"
    "https://github.com/1038lab/ComfyUI-RMBG"
    "https://github.com/spinagon/ComfyUI-seamless-tiling"
    "https://github.com/kijai/ComfyUI-segment-anything-2"
    "https://github.com/tanglaoya321/ComfyUI-StoryMaker"
    "https://github.com/ai-shizuka/ComfyUI-tbox"
    "https://github.com/un-seen/comfyui-tensorops"
    "https://github.com/Fannovel16/ComfyUI-Video-Matting"
    "https://github.com/gokayfem/ComfyUI_VLM_nodes"
    "https://github.com/yuvraj108c/ComfyUI-Whisper"
    "https://github.com/silveroxides/ComfyUI_bnb_nf4_fp4_Loaders"
    "https://github.com/cubiq/ComfyUI_InstantID"
    "https://github.com/chflame163/ComfyUI_LayerStyle"
    "https://github.com/smthemex/ComfyUI_MS_Diffusion"
    "https://github.com/lldacing/ComfyUI_PuLID_Flux_ll"
    "https://github.com/smthemex/ComfyUI_StoryDiffusion"
    "https://github.com/chaojie/ComfyUI-EasyAnimate"
    "https://github.com/bash-j/mikey_nodes"
    "https://github.com/cubiq/PuLID_ComfyUI"
    "https://github.com/SKBv0/ComfyUI_SKBundle"
    "https://github.com/Gourieff/ComfyUI-ReActor"
    "https://github.com/gokayfem/ComfyUI_VLM_nodes"
)

WORKFLOWS=(
    "https://gist.githubusercontent.com/robballantyne/f8cb692bdcd89c96c0bd1ec0c969d905/raw/2d969f732d7873f0e1ee23b2625b50f201c722a5/flux_dev_example.json"
)

# Models
LORA_MODELS=(
  #  "https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-Reward-LoRAs/resolve/main/CogVideoX-Fun-V1.1-5b-InP-MPS.safetensors:CogVideo/loras:CogVideoX-Fun-V1.1-5b-InP-MPS.safetensors"
    "https://huggingface.co/ali-vilab/ACE_Plus/resolve/main/portrait/comfyui_portrait_lora64.safetensors:loras/ACE:comfyui_portrait_lora64.safetensors"
    "https://huggingface.co/ali-vilab/ACE_Plus/resolve/main/subject/comfyui_subject_lora16.safetensors:loras/ACE:comfyui_subject_lora16.safetensors"
  # "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v_lora.safetensors:loras/SD15:AnimateLCM_sd15_t2v_lora.safetensors"
    "https://huggingface.co/xiaozaa/catvton-flux-lora-alpha/resolve/main/pytorch_lora_weights.safetensors:loras:catVtonLora.safetensors"
    "https://huggingface.co/enhanceaiteam/Flux-Uncensored-V2/resolve/main/lora.safetensors?download=true:loras/FLUX:uncensored-flux-lora.safetensors"
    "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-FLUX.1-dev-8steps-lora.safetensors:loras/FLUX/HYPERLFUX:Hyper-FLUX.1-dev-8steps-lora.safetensors"
    "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SD15-8steps-CFG-lora.safetensors:loras:Hyper-SD15-8steps-CFG-lora.safetensors"
    "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-12steps-CFG-lora.safetensors:loras:Hyper-SDXL-12steps-CFG-lora.safetensors"
    "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-8steps-CFG-lora.safetensors:loras:Hyper-SDXL-8steps-CFG-lora.safetensors"
    "https://civitai.com/api/download/models/129282:loras/SDXL:add_detail.safetensors"
    "https://civitai.com/api/download/models/78645:loras/SDXL:dreamScenery.safetensors"
    "https://civitai.com/api/download/models/157267:loras/SDXL:super_beauty_face.safetensors"
)

ANIMATEDIFF_MODELS=(
)

UNET_MODELS=(
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors"
    "https://huggingface.co/lllyasviel/flux1_dev/resolve/main/flux1-dev-fp8.safetensors:unet:flux1-dev-fp8.safetensors"
    "https://huggingface.co/jingheya/lotus-depth-g-v2-0-disparity/resolve/main/unet/diffusion_pytorch_model.safetensors:unet:lotus-depth-g-v2-0.safetensors"
)

UPSCALE_MODELS=(
    "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth:upscale_models:4x_foolhardy_Remacri.pth"
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth:upscale_models:RealESRGAN_x4plus.pth"
    "https://huggingface.co/datasets/neverjay/upscalers/resolve/main/4x-UltraSharp.pth:upscale_models:4x-UltraSharp.pth"
)

IPADAPTER_MODELS=(
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors:ipadapter:ip-adapter_sd15.safetensors"
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors:ipadapter:ip-adapter-plus_sd15.safetensors"
    "https://huggingface.co/h94/IP-Adapter-Plus-Face/resolve/main/ip-adapter-plus-face_sd15.safetensors:ipadapter:ip-adapter-plus-face_sd15.safetensors"
    "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors:ipadapter:ip-adapter_sdxl_vit-h.safetensors"
    "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors:ipadapter:ip-adapter-plus_sdxl_vit-h.safetensors"
    "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors:ipadapter:ip-adapter-plus-face_sdxl_vit-h.safetensors"
)

PULID_MODELS=(
    "https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors:pulid:pulid_flux_v0.9.1.safetensors"
    "https://huggingface.co/huchenlei/ipadapter_pulid/resolve/main/ip-adapter_pulid_sdxl_fp16.safetensors?download=true:pulid:ip-adapter_pulid_sdxl_fp16.safetensors"
)

INSTANTID_MODELS=(
    "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin?download=true:instantid:ip-adapter.bin"
)

PHOTOMAKER_MODELS=(
    "https://huggingface.co/TencentARC/PhotoMaker-V2/resolve/main/photomaker-v2.bin:photomaker:photomaker-v2.bin"
)

INPAINT_MODELS=(
    "https://huggingface.co/lllyasviel/mat/resolve/main/MAT_Places512_G_fp16.safetensors:inpaint:MAT_Places512_G_fp16.safetensors"
)

SAM_MODELS=(
    "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth:sams:sam_hq_vit_h.pth"
    "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/sams/sam_vit_b_01ec64.pth:sams:sam_vit_b_01ec64.pth"
)

LLM_MODELS=(
    "https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf:llm_gguf:Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
)

FACERESTORE_MODELS=(
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth:facerestore_models:GFPGANv1.3.pth"
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth:facerestore_models:GFPGANv1.4.pth"
)

VAE_APPROX_MODELS=(
    "https://github.com/madebyollin/taesd/raw/main/taesd_decoder.pth:vae_approx:taesd_decoder.pth"
    "https://github.com/madebyollin/taesd/raw/main/taesd_encoder.pth:vae_approx:taesd_encoder.pth"
    "https://github.com/madebyollin/taesd/raw/main/taesdxl_decoder.pth:vae_approx:taesdxl_decoder.pth"
    "https://github.com/madebyollin/taesd/raw/main/taesdxl_encoder.pth:vae_approx:taesdxl_encoder.pth"
)

DIFFUSION_MODELS=(
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors:diffusion_models:wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors"
    "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors:diffusion_models/IC-Light:iclight_sd15_fc.safetensors"
    "https://huggingface.co/lllyasviel/flux1-fill/resolve/main/flux1-fill-dev.safetensors:diffusion_models:flux1-fill-dev.safetensors"
)

CONTROLNET_MODELS=(
    "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/diffusion_pytorch_model.safetensors:controlnet:Flux_contolnet_union.safetensors"
    "https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/control_v1p_sd15_qrcode_monster.safetensors:controlnet/animatediff:control_v1p_sd15_qrcode_monster.safetensors"
    "https://huggingface.co/monster-labs/control_v1p_sdxl_qrcode_monster/resolve/main/diffusion_pytorch_model.safetensors:controlnet:control_v1p_sdxl_qrcode_monster.safetensors"
    "https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/resolve/main/diffusion_pytorch_model.fp16.safetensors:controlnet:control_v11f1p_sd15_depth.safetensors"
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth:controlnet:control_v11f1e_sd15_tile.pth"
    "https://huggingface.co/lllyasviel/control_v11p_sd15_lineart/resolve/main/diffusion_pytorch_model.safetensors:controlnet:control_v11p_sd15_lineart.safetensors"
    "https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/diffusion_pytorch_model.safetensors:controlnet:control_v11p_sd15_openpose.safetensors"
    "https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors:controlnet:controlnet-union-sdxl-1.0.safetensors"
    "https://huggingface.co/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta/resolve/main/diffusion_pytorch_model.safetensors:controlnet:FLUX.1-dev-Controlnet-Inpainting-Beta.safetensors"
    "https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors?download=true:controlnet:diffusion_pytorch_model.safetensors"
    "https://huggingface.co/thibaud/controlnetxs/resolve/main/control_openpose_xl_1_0.safetensors:controlnet:Control_OpenPoseXL2.safetensors"
    "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors:controlnet:controlnet-cany-sdxl-1.0fp16.safetensors"
    "https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors:controlnet:controlnet-depth-sdxl-1.0fp16.safetensors"
    "https://huggingface.co/xinsir/controlnet-tile-xl-12dt/resolve/main/controlnet-tile-xl-12dt.safetensors:controlnet/XL:XL_TILE_XINSIR.safetensors"
    "https://huggingface.co/stabilityai/control-lora/resolve/main/control-loras-canny/control-lora-canny-rank128.safetensors:controlnet/SDXL:control-lora-canny-rank128.safetensors"
    "https://huggingface.co/stabilityai/control-lora/resolve/main/control-loras-canny/control-lora-canny-rank256.safetensors:controlnet/SDXL:control-lora-canny-rank256.safetensors"
    "https://huggingface.co/stabilityai/control-lora/resolve/main/control-loras-depth/control-lora-depth-rank256.safetensors:controlnet/XL:control-lora-depth-rank256.safetensors"
    "https://huggingface.co/thibaud/controlnetxs/resolve/main/control-lora-openposeXL2-rank256.safetensors:controlnet/XL:control-lora-openposeXL2-rank256.safetensors"
    "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Depth/resolve/main/diffusion_pytorch_model.safetensors:controlnet/FLUX.1/Shakker-Labs-ControlNet-Union-Pro:FLUX.1-dev-ControlNet-Depth.safetensors"
    "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/diffusion_pytorch_model.safetensors:controlnet/FLUX.1/Shakker-Labs-ControlNet-Union-Pro:FLUX.1-dev-ControlNet-Union-Pro.safetensors"
)

CLIP_VISION_MODELS=(
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors:clip_vision:clip_vision_h.safetensors"
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors:clip_vision:clipvision-for-ip-adapter.safetensors"
)

VAE_MODELS=(
  # "https://huggingface.co/comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI/resolve/main/vae/cosmos_cv8x8x8_1.0.safetensors:vae:cosmos_cv8x8x8_1.0.safetensors"
  # "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors:vae:wan_2.1_vae.safetensors"
    "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/vae/diffusion_pytorch_model.safetensors:vae/FLUX1:ae.safetensors"
    "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors:vae:vae-ft-mse-840000-ema-pruned.safetensors"
    "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors:vae:sdxl.vae.safetensors"
  # "https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/vae/pytorch_model.pt:vae:vae-hunyuan.pt"
)

CHECKPOINT_MODELS=(
    "https://huggingface.co/Lykon/AbsoluteReality/resolve/main/AbsoluteReality_1.8.1_pruned.safetensors:checkpoints2:AbsoluteReality_1.8.1_pruned.safetensors"
    "https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/resolve/main/flux1-dev-bnb-nf4.safetensors:checkpoints2:flux1-dev-bnb-nf4.safetensors"
    "https://huggingface.co/alexgenovese/reica_models/resolve/021e192bd744c48a85f8ae1832662e77beb9aac7/realvisxlV40_v40LightningBakedvae.safetensors:checkpoints2:realvisxlV40_v40LightningBakedvae.safetensors"
    "https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/sd3.5_large_fp8_scaled.safetensors:checkpoints2:sd3.5_large_fp8_scaled.safetensors"
    "https://huggingface.co/Kijai/SUPIR_pruned/resolve/main/SUPIR-v0F_fp16.safetensors:checkpoints2:SUPIR-v0F_fp16.safetensors"
    "https://civitai.com/api/download/models/169921:checkpoints2:juggernautXL_v9Rdphoto2Lightning.safetensors"
    "https://huggingface.co/lllyasviel/flux1-dev-compact/resolve/main/flux1CompactCLIPAnd_Flux1DevFp8.safetensors:checkpoints2:flux1CompactCLIPAnd_Flux1DevFp8.safetensors"
    "https://civitai.com/api/download/models/149281:checkpoints2:truesketchsdxl_v10.safetensors"
    "https://civitai.com/api/download/models/133032:checkpoints2:epicphotogasm_ultimateFidelity.safetensors"
)

STYLE_MODELS=(
    "https://huggingface.co/lllyasviel/flux1-redux/resolve/main/flux1-redux-dev.safetensors:style_models:flux1-redux-dev.safetensors"
)

LAYER_MODELS=(
    "https://huggingface.co/layerdiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_attn.safetensors:layer_model:layer_xl_transparent_attn.safetensors"
    "https://huggingface.co/layerdiffusion/layerdiffusion-v1/resolve/main/vae_transparent_decoder.safetensors:layer_model:vae_transparent_decoder.safetensors"
)

RMBG_MODELS=(
    "https://huggingface.co/uwg/modelscope/resolve/main/InSPyReNet/models/inspyrenet.pth:RMBG/INSPYRENET:inspyrenet.safetensors"
)

INSIGHTFACE_MODELS=(
    "https://github.com/deepinsight/insightface/raw/master/python-package/insightface/models/antelopev2/1k3d68.onnx:insightface/models/antelopev2:1k3d68.onnx"
    "https://github.com/deepinsight/insightface/raw/master/python-package/insightface/models/antelopev2/2d106det.onnx:insightface/models/antelopev2:2d106det.onnx"
    "https://github.com/deepinsight/insightface/raw/master/python-package/insightface/models/antelopev2/genderage.onnx:insightface/models/antelopev2:genderage.onnx"
    "https://github.com/deepinsight/insightface/raw/master/python-package/insightface/models/antelopev2/scrfd_10g_bnkps.onnx:insightface/models/antelopev2:scrfd_10g_bnkps.onnx"
)

FACEDETECTION_MODELS=(
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth:facedetection:detection_Resnet50_Final.pth"
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/parsing_parsenet.pth:facedetection:parsing_parsenet.pth"
    "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2:facedetection:shape_predictor_68_face_landmarks.dat.bz2"
)

ULTRALYTICS_MODELS=(
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-face.pt:ultralytics/bbox:face_yolov8m.pt"
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-hand.pt:ultralytics/bbox:hand_yolov8s.pt"
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt:ultralytics/segm:person_yolov8m-seg.pt"
)

CLIP_MODELS=(
    "https://huggingface.co/comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI/resolve/main/text_encoders/oldt5_xxl_fp8_e4m3fn_scaled.safetensors:clip:oldt5_xxl_fp8_e4m3fn_scaled.safetensors"
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors:clip:t5xxl_fp8_e4m3fn.safetensors"
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors:clip:clip_l-for-gguf.safetensors"
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors:clip:umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin:clip:clip_l.safetensors"
    "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/llava_llama3_fp8_scaled.safetensors:clip:llava_llama3_fp8_scaled.safetensors"
)

# Main execution
if [ ! -f /.noprovisioning ]; then
    provisioning_start
fi