import os
from llava.model.multimodal_encoder.medical_vit_encoder import MedicalViTTower
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower, CLIPVisionTowerS2


def build_vision_tower(vision_tower_cfg, **kwargs):
    """Build the vision encoder"""
    
    vision_tower = getattr(
        vision_tower_cfg, 
        'mm_vision_tower', 
        getattr(vision_tower_cfg, 'vision_tower', None)
    )
    
    # Check if it's your medical ViT
    if vision_tower and 'medical_vit' in vision_tower.lower():
        print("Building Medical ViT Tower...")
        
        
        # Get checkpoint path from config
        checkpoint_path = getattr(
            vision_tower_cfg,
            'vit_checkpoint_path',
            None
        )
        
        if checkpoint_path is None:
            raise ValueError("vit_checkpoint_path must be provided for medical_vit")
        
        # Get token strategy
        use_all_tokens = getattr(
            vision_tower_cfg,
            'use_all_vit_tokens',
            True  # Default to using all tokens
        )
        
        return MedicalViTTower(
            vit_checkpoint_path=checkpoint_path,
            use_all_tokens=use_all_tokens
        )
    
    # Original CLIP handling (keep this logic intact)
    is_absolute_path_exists = os.path.exists(vision_tower) if vision_tower else False
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    raise ValueError(f'Unknown vision tower: {vision_tower}')