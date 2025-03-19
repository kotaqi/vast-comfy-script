function provisioning_get_models() {
    # Define an array of model categories to download
    local MODEL_CATEGORIES=(
        "CHECKPOINT_MODELS:checkpoints"
        "VAE_MODELS:vae"
        "CLIP_MODELS:clip"
        "CLIP_VISION_MODELS:clip_vision"
        "CONTROLNET_MODELS:controlnet"
        "DIFFUSION_MODELS:diffusion_models"
        "LORA_MODELS:loras"
        "ANIMATEDIFF_MODELS:animatediff_models"
        "UNET_MODELS:unet"
        "UPSCALE_MODELS:upscale_models"
        "IPADAPTER_MODELS:ipadapter"
        "PULID_MODELS:pulid"
        "INSTANTID_MODELS:instantid"
        "PHOTOMAKER_MODELS:photomaker"
        "INPAINT_MODELS:inpaint"
        "SAM_MODELS:sams"
        "LLM_MODELS:llm_gguf"
        "FACERESTORE_MODELS:facerestore_models"
        "VAE_APPROX_MODELS:vae_approx"
    )

    # Iterate through each model category
    for category in "${MODEL_CATEGORIES[@]}"; do
        IFS=':' read -r model_var default_dir <<< "$category"
        
        # Use indirect variable reference to get the model list
        models_to_download=($(eval echo \${${model_var}[@]}))
        
        # Skip if no models in this category
        if [ ${#models_to_download[@]} -eq 0 ]; then
            continue
        fi
        
        echo "Downloading ${model_var} to ${default_dir}..."
        
        # Download each model in the category
        for model in "${models_to_download[@]}"; do
            IFS=':' read -r url subdir filename <<< "$model"
            
            # Use subdirectory if specified, otherwise use default
            target_dir="${COMFYUI_DIR}/models/${subdir:-$default_dir}"
            
            # Create target directory if it doesn't exist
            mkdir -p "$target_dir"
            
            # Download the model
            provisioning_download_with_retry "$url" "$target_dir" "$filename"
        done
    done
}ANIMATEDIFF_MODELS=(
    "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt:animatediff_models:mm_sd_v15_v2.ckpt"
    "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v.ckpt:animatediff_models:AnimateLCM_sd15_t2v.ckpt"
)

UNET_MODELS=(
    "https://huggingface.co/lllyasviel/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q4_0.gguf:unet:flux1-dev-Q4_0.gguf"
    "https://huggingface.co/lllyasviel/flux1_dev/resolve/main/flux1-dev-fp8.safetensors:unet:flux1-dev-fp8.safetensors"
    "https://huggingface.co/jingheya/lotus-normal-d-v1-0/resolve/main/unet/diffusion_pytorch_model.safetensors:unet:lotus-normal-d-v1-0.safetensors"
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
    "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth:inpaint:fooocus_inpaint_head.pth"
    "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/8fcfd208b8e76537f23ae061dc3e3d26714ee4ec/inpaint_v26.fooocus.patch:inpaint:inpaint_v26.fooocus.patch"
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
)DIFFUSION_MODELS=(
    "https://huggingface.co/mcmonkey/cosmos-1.0/resolve/main/Cosmos-1_0-Diffusion-7B-Text2World.safetensors:diffusion_models:Cosmos-1_0-Diffusion-7B-Text2World.safetensors"
    "https://huggingface.co/mcmonkey/cosmos-1.0/resolve/main/Cosmos-1_0-Diffusion-7B-Video2World.safetensors:diffusion_models:Cosmos-1_0-Diffusion-7B-Video2World.safetensors"
    "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/diffusion_models/hunyuan_video_t2v_720p_bf16.safetensors:diffusion_models:hunyuan_video_t2v_720p_bf16.safetensors"
    "https://huggingface.co/Kijai/SkyReels-V1-Hunyuan_comfy/resolve/main/skyreels_hunyuan_i2v_fp8_e4m3fn.safetensors:diffusion_models:skyreels_hunyuan_i2v_fp8_e4m3fn.safetensors"
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors:diffusion_models:wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors"
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_720p_14B_fp8_e4m3fn.safetensors:diffusion_models:wan2.1_i2v_720p_14B_fp8_e4m3fn.safetensors"
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors:diffusion_models:wan2.1_t2v_1.3B_bf16.safetensors"
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14B_fp8_e4m3fn.safetensors:diffusion_models:wan2.1_t2v_14B_fp8_e4m3fn.safetensors"
    "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors:diffusion_models/IC-Light:iclight_sd15_fc.safetensors"
)

LORA_MODELS=(
    "https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-Reward-LoRAs/resolve/main/CogVideoX-Fun-V1.1-5b-InP-MPS.safetensors:CogVideo/loras:CogVideoX-Fun-V1.1-5b-InP-MPS.safetensors"
    "https://huggingface.co/ali-vilab/ACE_Plus/resolve/main/portrait/comfyui_portrait_lora64.safetensors:loras/ACE:comfyui_portrait_lora64.safetensors"
    "https://huggingface.co/ali-vilab/ACE_Plus/resolve/main/subject/comfyui_subject_lora16.safetensors:loras/ACE:comfyui_subject_lora16.safetensors"
    "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v_lora.safetensors:loras/SD15:AnimateLCM_sd15_t2v_lora.safetensors"
    "https://huggingface.co/xiaozaa/catvton-flux-lora-alpha/resolve/main/pytorch_lora_weights.safetensors:loras:catVtonLora.safetensors"
    "https://huggingface.co/enhanceaiteam/Flux-Uncensored-V2/resolve/main/lora.safetensors?download=true:loras/FLUX:uncensored-flux-lora.safetensors"
    "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-FLUX.1-dev-8steps-lora.safetensors:loras/FLUX/HYPERLFUX:Hyper-FLUX.1-dev-8steps-lora.safetensors"
    "https://huggingface.co/spacepxl/skyreels-i2v-smooth-lora/resolve/main/skyreels-i2v-smooth-lora-test-00000350.safetensors:loras:skyreels-i2v-smooth-lora-test-00000350.safetensors"
)CLIP_MODELS=(
    "https://huggingface.co/comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI/resolve/main/text_encoders/oldt5_xxl_fp8_e4m3fn_scaled.safetensors:clip:oldt5_xxl_fp8_e4m3fn_scaled.safetensors"
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors:clip:t5xxl_fp8_e4m3fn.safetensors"
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors:clip:clip_l-for-gguf.safetensors"
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors:clip:umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin:clip:clip_l.safetensors"
    "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/llava_llama3_fp8_scaled.safetensors:clip:llava_llama3_fp8_scaled.safetensors"
)

CLIP_VISION_MODELS=(
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors:clip_vision:clip_vision_h.safetensors"
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors:clip_vision:clipvision-for-ip-adapter.safetensors"
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
)VAE_MODELS=(
    "https://huggingface.co/comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI/resolve/main/vae/cosmos_cv8x8x8_1.0.safetensors:vae:cosmos_cv8x8x8_1.0.safetensors"
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors:vae:wan_2.1_vae.safetensors"
    "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/vae/diffusion_pytorch_model.safetensors:vae/FLUX1:ae.safetensors"
    "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors:vae:vae-ft-mse-840000-ema-pruned.safetensors"
    "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors:vae:sdxl.vae.safetensors"
    "https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/vae/pytorch_model.pt:vae:vae-hunyuan.pt"
)# Comprehensive model download lists
CHECKPOINT_MODELS=(
    "https://huggingface.co/Lykon/AbsoluteReality/resolve/main/AbsoluteReality_1.8.1_pruned.safetensors:checkpoints:AbsoluteReality_1.8.1_pruned.safetensors"
    "https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/resolve/main/flux1-dev-bnb-nf4.safetensors:checkpoints:flux1-dev-bnb-nf4.safetensors"
    "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.5.safetensors:checkpoints:ltx-video-2b-v0.9.5.safetensors"
    "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors:checkpoints:svd_xt.safetensors"
    "https://huggingface.co/alexgenovese/reica_models/resolve/021e192bd744c48a85f8ae1832662e77beb9aac7/realvisxlV40_v40LightningBakedvae.safetensors:checkpoints:realvisxlV40_v40LightningBakedvae.safetensors"
    "https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/sd3.5_large_fp8_scaled.safetensors:checkpoints:sd3.5_large_fp8_scaled.safetensors"
    "https://huggingface.co/Kijai/SUPIR_pruned/resolve/main/SUPIR-v0F_fp16.safetensors:checkpoints:SUPIR-v0F_fp16.safetensors"
)function provisioning_start() {
    provisioning_print_header
    
    # Create ComfyUI directory
    mkdir -p "${COMFYUI_DIR}"
    cd "${COMFYUI_DIR}"

    # Install base requirements
    provisioning_get_apt_packages
    provisioning_get_pip_packages

    # Clone ComfyUI main repository if not exists
    if [[ ! -d "${COMFYUI_DIR}" ]]; then
        git clone https://github.com/comfyanonymous/ComfyUI.git .
        pip install -r requirements.txt
    fi

    # Create custom nodes directory
    mkdir -p "${COMFYUI_DIR}/custom_nodes"
    
    # Download and install custom nodes FIRST
    provisioning_get_nodes
    
    # Create workflows directory
    workflows_dir="${COMFYUI_DIR}/user/default/workflows"
    mkdir -p "${workflows_dir}"
    
    # Download workflows
    provisioning_get_files \
        "${workflows_dir}" \
        "${WORKFLOWS[@]}"
    
    # Download models AFTER nodes are in place
    provisioning_get_models
    
    # Get licensed models if HF_TOKEN set & valid
    if provisioning_has_valid_hf_token; then
        UNET_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors")
        VAE_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors")
    else
        UNET_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors")
        VAE_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors")
        sed -i 's/flux1-dev\.safetensors/flux1-schnell.safetensors/g' "${workflows_dir}/flux_dev_example.json"
    fi
    
    # Download additional UNET and VAE models
    provisioning_get_files \
        "${COMFYUI_DIR}/models/unet" \
        "${UNET_MODELS[@]}"
    provisioning_get_files \
        "${COMFYUI_DIR}/models/vae" \
        "${VAE_MODELS[@]}"
    
    provisioning_print_end
}# Model download lists with comprehensive URLs
CHECKPOINT_MODELS=(
    "https://huggingface.co/Lykon/AbsoluteReality/resolve/main/AbsoluteReality_1.8.1_pruned.safetensors:checkpoints:AbsoluteReality_1.8.1_pruned.safetensors"
    "https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/resolve/main/flux1-dev-bnb-nf4.safetensors:checkpoints:flux1-dev-bnb-nf4.safetensors"
    "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.5.safetensors:checkpoints:ltx-video-2b-v0.9.5.safetensors"
    "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors:checkpoints:svd_xt.safetensors"
    "https://huggingface.co/alexgenovese/reica_models/resolve/021e192bd744c48a85f8ae1832662e77beb9aac7/realvisxlV40_v40LightningBakedvae.safetensors:checkpoints:realvisxlV40_v40LightningBakedvae.safetensors"
    "https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/sd3.5_large_fp8_scaled.safetensors:checkpoints:sd3.5_large_fp8_scaled.safetensors"
    "https://huggingface.co/Kijai/SUPIR_pruned/resolve/main/SUPIR-v0F_fp16.safetensors:checkpoints:SUPIR-v0F_fp16.safetensors"
)

VAE_MODELS=(
    "https://huggingface.co/comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI/resolve/main/vae/cosmos_cv8x8x8_1.0.safetensors:vae:cosmos_cv8x8x8_1.0.safetensors"
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors:vae:wan_2.1_vae.safetensors"
    "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/vae/diffusion_pytorch_model.safetensors:vae/FLUX1:ae.safetensors"
    "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors:vae:vae-ft-mse-840000-ema-pruned.safetensors"
    "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors:vae:sdxl.vae.safetensors"
    "https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/vae/pytorch_model.pt:vae:vae-hunyuan.pt"
)

CLIP_MODELS=(
    "https://huggingface.co/comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI/resolve/main/text_encoders/oldt5_xxl_fp8_e4m3fn_scaled.safetensors:clip:oldt5_xxl_fp8_e4m3fn_scaled.safetensors"
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors:clip:t5xxl_fp8_e4m3fn.safetensors"
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors:clip:clip_l-for-gguf.safetensors"
    "https://huggingface.NODES=(
    "https://github.com/comfyanonymous/ComfyUI-Canvas-Tab"
    "https://github.com/chrisgoringe/cg-use-everywhere"
    "https://github.com/ZHO-ZHO-ZHO/ComfyUI-CharacterFaceSwap"
    "https://github.com/AiSearch/comfy-image-saver"
    "https://github.com/chrisgoringe/ComfyLiterals"
    "https://github.com/chrisgoringe/ComfyMath"
    "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet"
    "https://github.com/fishslot/ComfyUI-AdvancedLivePortrait"
    "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved"
    "https://github.com/artventure/comfyui-art-venture"
    "https://github.com/FlyingFireCo/comfyui-autocropfaces"
    "https://github.com/0xbitches/comfyui-automaticcfg"
    "https://github.com/TencentARC/ComfyUI-Bringing-Old-Photos-Back-to-Life"
    "https://github.com/whitebrowserx/ComfyUI-Chat-GPT-Integration"
    "https://github.com/RockOfFire/ComfyUI-Chibi-Nodes"
    "https://github.com/xianyuntang/ComfyUI-CogVideoXWrapper"
    "https://github.com/crystian/ComfyUI-Crystools"
    "https://github.com/congsongxing/ComfyUI-CSV-Loader"
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"
    "https://github.com/Jcd1230/comfyui-dduf"
    "https://github.com/ChenZuohui/ComfyUI-Depthflow-Nodes"
    "https://github.com/huggingface/comfyui-diffusers"
    "https://github.com/senshilabs/ComfyUI-Easy-Use"
    "https://github.com/TheWDeveloper/ComfyUI-F5-TTS"
    "https://github.com/AzAIArtificial/comfyui-fitsize"
    "https://github.com/SkunkworksAI/ComfyUI-Florence2"
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"
    "https://github.com/ZsiroskenyerA/ComfyUI-GGUF"
    "https://github.com/griptape-ai/comfyui-griptape"
    "https://github.com/yonghengg/ComfyUI-HunyuanlVideoWrapper"
    "https://github.com/ImprUve/ComfyUI-IC-Light"
    "https://github.com/ImprUve/ComfyUI-IC-Light-Native"
    "https://github.com/Haoming02/ComfyUI-Image-Filters"
    "https://github.com/ltdrdata/ComfyUI-Impact-Pack"
    "https://github.com/ltdrdata/comfyui-impact-subpack"
    "https://github.com/mikeymike94/ComfyUI-Inpaint-CropAndStitch"
    "https://github.com/Fannovel16/comfyui-inpaint-nodes"
    "https://github.com/ltdrdata/ComfyUI-Inspire-Pack"
    "https://github.com/dunkeroni/ComfyUI-JDCN"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/wuyi2020/comfyui-lama-remover"
    "https://github.com/shushengboosix/ComfyUI-LatentSyncWrapper"
    "https://github.com/layerdiffusion/sd-forge-layerdiffuse"
    "https://github.com/umoho/ComfyUI-Lotus"
    "https://github.com/kaino46/ComfyUI-LTXVideo"
    "https://github.com/ltdrdata/ComfyUI-Manager"
    "https://github.com/bdsqlsz/ComfyUI-Marigold"
    "https://github.com/nixingyang/ComfyUI-MixLab-Nodes"
    "https://github.com/garryye/comfyui-mvadapter"
    "https://github.com/shoriful-islam-shorif/comfyui-ollama"
    "https://github.com/ZHO-ZHO-ZHO/ComfyUI-PhotoMaker-Plus"
    "https://github.com/ZHO-ZHO-ZHO/ComfyUI-PhotoMaker-ZHO"
    "https://github.com/florestefano1975/comfyui-portrait-master"
    "https://github.com/theOGpython/ComfyUI-Prompt-Combinator"
    "https://github.com/libcxu/ComfyUI-PuLID-Flux-Enhanced"
    "https://github.com/HorizonPower/ComfyUI-PyramidFlowWrapper"
    "https://github.com/Omar-Fahmi/ComfyUI-QualityOfLifeSuit_Omar92"
    "https://github.com/Gourieff/comfyui-reactor-node"
    "https://github.com/hnmr293/ComfyUI-ResAdapter"
    "https://github.com/ZHO-ZHO-ZHO/ComfyUI-RMBG"
    "https://github.com/ThanThoai/ComfyUI-seamless-tiling"
    "https://github.com/seeyouit/ComfyUI-segment-anything-2"
    "https://github.com/Stability-AI/ComfyUI-StoryMaker"
    "https://github.com/pythongosssss/ComfyUI-Styles_CSV_Loader"
    "https://github.com/ChenWu98/ComfyUI-SUPIR"
    "https://github.com/talesofai/comfyui-tbox"
    "https://github.com/cheetman/ComfyUI-TeaCache"
    "https://github.com/zapmuj/comfyui-tensorops"
    "https://github.com/Acly/comfyui-tooling-nodes"
    "https://github.com/krekeltronics/ComfyUI-ToSVG"
    "https://github.com/diontimmer/ComfyUI-Various-Nodes"
    "https://github.com/mldz/comfyui-video-matting"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/Nuked88/ComfyUI-VLM-Captions"
    "https://github.com/biegert/ComfyUI-Whisper"
    "https://github.com/WASasquatch/ComfyUI-WildPromptor"
    "https://github.com/chrisgoringe/ComfyUI_AdvancedRefluxControl"
    "https://github.com/shiimizu/ComfyUI_bnb_nf4_fp4_Loaders"
    "https://github.com/Comfyroll/ComfyUI_Comfyroll_CustomNodes"
    "https://github.com/Fannovel16/comfyui_controlnet_aux"
    "https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet"
    "https://github.com/cubiq/ComfyUI_essentials"
    "https://github.com/comfyanonymous/ComfyUI_experiments"
    "https://github.com/FizzleDorf/ComfyUI_FizzNodes"
    "https://github.com/aihey/ComfyUI_InstantID"
    "https://github.com/cubiq/ComfyUI_IPAdapter_plus"
    "https://github.com/diontimmer/ComfyUI_LayerStyle"
    "https://github.com/a1lazydog/ComfyUI_LayerStyle_Advance"
    "https://github.com/Taishi-Kuribara/ComfyUI_Mira"
    "https://github.com/ashawkey/comfyui_ms_diffusion"
    "https://github.com/LEv145/ComfyUI_Patches_ll"
    "https://github.com/libcxu/ComfyUI_PuLID_Flux_ll"
    "https://github.com/SeargeDP/ComfyUI_Searge_LLM"
    "https://github.com/storyicon/comfyui_segment_anything"
    "https://github.com/mmanar/ComfyUI_smZNodes"
    "https://github.com/huchenlei/ComfyUI_Sonic"
    "https://github.com/Visionary/comfyui_storydiffusion"
    "https://github.com/TinyTerra/ComfyUI_tinyterraNodes"
    "https://github.com/ssitu/ComfyUI_UltimateSDUpscale"
    "https://github.com/ArtVentureX/comfyui-easyanimate"
    "https://github.com/NullID/mikey_nodes"
    "https://github.com/chrisgoringe/primitive-types"
    "https://github.com/averad/PuLID_ComfyUI"
    "https://github.com/rgthree/rgthree-comfy"
    "https://github.com/Ruyi-Cyber-Lab/Ruyi-Models"
    "https://github.com/mcmonkeyprojects/sd-dynamic-thresholding"
    "https://github.com/BlenderNeko/ComfyUI_TiledKSampler"
    "https://github.com/space-nuko/ComfyUI-skBUNDLE"
    "https://github.com/was-node-suite-comfyui"
)#!/bin/bash

source /venv/main/bin/activate
COMFYUI_DIR=${WORKSPACE}/ComfyUI

# Function for downloading with retry logic
function provisioning_download_with_retry() {
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

# Additional packages you want to install
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

# Custom nodes you want to add
NODES=(
    "https://github.com/ltdrdata/ComfyUI-Manager"
    "https://github.com/comfyanonymous/ComfyUI-Canvas-Tab"
    "https://github.com/chrisgoringe/cg-use-everywhere"
    "https://github.com/ZHO-ZHO-ZHO/ComfyUI-CharacterFaceSwap"
    "https://github.com/AiSearch/comfy-image-saver"
    "https://github.com/chrisgoringe/ComfyLiterals"
    "https://github.com/chrisgoringe/ComfyMath"
    "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet"
    "https://github.com/fishslot/ComfyUI-AdvancedLivePortrait"
    "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved"
    "https://github.com/artventure/comfyui-art-venture"
    "https://github.com/FlyingFireCo/comfyui-autocropfaces"
    "https://github.com/0xbitches/comfyui-automaticcfg"
)

# Workflows to download
WORKFLOWS=(
    "https://gist.githubusercontent.com/robballantyne/f8cb692bdcd89c96c0bd1ec0c969d905/raw/2d969f732d7873f0e1ee23b2625b50f201c722a5/flux_dev_example.json"
)

# Define model download lists by category
# Note: You might want to comment out or remove URLs for models you don't need or want to download
CHECKPOINT_MODELS=(
    "https://huggingface.co/Lykon/AbsoluteReality/resolve/main/AbsoluteReality_1.8.1_pruned.safetensors:checkpoints:AbsoluteReality_1.8.1_pruned.safetensors"
    "https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/resolve/main/flux1-dev-bnb-nf4.safetensors:checkpoints:flux1-dev-bnb-nf4.safetensors"
    "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.5.safetensors:checkpoints:ltx-video-2b-v0.9.5.safetensors"
)

VAE_MODELS=(
    "https://huggingface.co/comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI/resolve/main/vae/cosmos_cv8x8x8_1.0.safetensors:vae:cosmos_cv8x8x8_1.0.safetensors"
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors:vae:wan_2.1_vae.safetensors"
    "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/vae/diffusion_pytorch_model.safetensors:vae/FLUX1:ae.safetensors"
)

CLIP_MODELS=(
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors:clip:clip_l-for-gguf.safetensors"
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors:clip:t5xxl_fp16.safetensors"
)

# Additional model categories you might want to download
UNET_MODELS=()
CONTROLNET_MODELS=()
LORA_MODELS=()
UPSCALE_MODELS=()

### DO NOT EDIT BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###

function provisioning_get_models() {
    # Download checkpoint models
    for model in "${CHECKPOINT_MODELS[@]}"; do
        IFS=':' read -r url subdir filename <<< "$model"
        provisioning_download_with_retry "$url" "${COMFYUI_DIR}/models/${subdir}" "$filename"
    done

    # Download VAE models
    for model in "${VAE_MODELS[@]}"; do
        IFS=':' read -r url subdir filename <<< "$model"
        provisioning_download_with_retry "$url" "${COMFYUI_DIR}/models/${subdir}" "$filename"
    done

    # Download CLIP models
    for model in "${CLIP_MODELS[@]}"; do
        IFS=':' read -r url subdir filename <<< "$model"
        provisioning_download_with_retry "$url" "${COMFYUI_DIR}/models/${subdir}" "$filename"
    done

    # You can add similar logic for other model types (UNET, CONTROLNET, etc.)
}

function provisioning_start() {
    provisioning_print_header
    provisioning_get_apt_packages
    provisioning_get_nodes
    provisioning_get_pip_packages
    
    # Create workflows directory
    workflows_dir="${COMFYUI_DIR}/user/default/workflows"
    mkdir -p "${workflows_dir}"
    
    # Download workflows
    provisioning_get_files \
        "${workflows_dir}" \
        "${WORKFLOWS[@]}"
    
    # Download models
    provisioning_get_models
    
    # Get licensed models if HF_TOKEN set & valid
    if provisioning_has_valid_hf_token; then
        UNET_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors")
        VAE_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors")
    else
        UNET_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors")
        VAE_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors")
        sed -i 's/flux1-dev\.safetensors/flux1-schnell.safetensors/g' "${workflows_dir}/flux_dev_example.json"
    fi
    
    # Download additional UNET and VAE models
    provisioning_get_files \
        "${COMFYUI_DIR}/models/unet" \
        "${UNET_MODELS[@]}"
    provisioning_get_files \
        "${COMFYUI_DIR}/models/vae" \
        "${VAE_MODELS[@]}"
    
    provisioning_print_end
}

# Rest of the script remains the same as in the previous version
# (functions like provisioning_get_apt_packages, provisioning_get_nodes, etc.)

function provisioning_get_apt_packages() {
    if [[ -n $APT_PACKAGES ]]; then
            sudo $APT_INSTALL ${APT_PACKAGES[@]}
    fi
}

function provisioning_get_pip_packages() {
    if [[ -n $PIP_PACKAGES ]]; then
            pip install --no-cache-dir ${PIP_PACKAGES[@]}
    fi
}

function provisioning_get_nodes() {
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="${COMFYUI_DIR}custom_nodes/${dir}"
        requirements="${path}/requirements.txt"
        if [[ -d $path ]]; then
            if [[ ${AUTO_UPDATE,,} != "false" ]]; then
                printf "Updating node: %s...\n" "${repo}"
                ( cd "$path" && git pull )
                if [[ -e $requirements ]]; then
                   pip install --no-cache-dir -r "$requirements"
                fi
            fi
        else
            printf "Downloading node: %s...\n" "${repo}"
            git clone "${repo}" "${path}" --recursive
            if [[ -e $requirements ]]; then
                pip install --no-cache-dir -r "${requirements}"
            fi
        fi
    done
}

function provisioning_get_files() {
    if [[ -z $2 ]]; then return 1; fi
    
    dir="$1"
    mkdir -p "$dir"
    shift
    arr=("$@")
    printf "Downloading %s model(s) to %s...\n" "${#arr[@]}" "$dir"
    for url in "${arr[@]}"; do
        printf "Downloading: %s\n" "${url}"
        provisioning_download "${url}" "${dir}"
        printf "\n"
    done
}

function provisioning_print_header() {
    printf "\n##############################################\n#                                            #\n#          Provisioning container            #\n#                                            #\n#         This will take some time           #\n#                                            #\n# Your container will be ready on completion #\n#                                            #\n##############################################\n\n"
}

function provisioning_print_end() {
    printf "\nProvisioning complete:  Application will start now\n\n"
}

function provisioning_has_valid_hf_token() {
    [[ -n "$HF_TOKEN" ]] || return 1
    url="https://huggingface.co/api/whoami-v2"

    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $HF_TOKEN" \
        -H "Content-Type: application/json")

    # Check if the token is valid
    if [ "$response" -eq 200 ]; then
        return 0
    else
        return 1
    fi
}

function provisioning_has_valid_civitai_token() {
    [[ -n "$CIVITAI_TOKEN" ]] || return 1
    url="https://civitai.com/api/v1/models?hidden=1&limit=1"

    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $CIVITAI_TOKEN" \
        -H "Content-Type: application/json")

    # Check if the token is valid
    if [ "$response" -eq 200 ]; then
        return 0
    else
        return 1
    fi
}

# Download from $1 URL to $2 file path
function provisioning_download() {
    if [[ -n $HF_TOKEN && $1 =~ ^https://([a-zA-Z0-9_-]+\.)?huggingface\.co(/|$|\?) ]]; then
        auth_token="$HF_TOKEN"
    elif 
        [[ -n $CIVITAI_TOKEN && $1 =~ ^https://([a-zA-Z0-9_-]+\.)?civitai\.com(/|$|\?) ]]; then
        auth_token="$CIVITAI_TOKEN"
    fi
    if [[ -n $auth_token ]];then
        wget --header="Authorization: Bearer $auth_token" -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$2" "$1"
    else
        wget -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$2" "$1"
    fi
}

# Allow user to disable provisioning if they started with a script they didn't want
if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi