import torch
import torch.nn as nn
import timm

class TIMMImageProcessor:
    def __init__(self, vision_tower):
        self.image_mean = list(vision_tower.default_cfg["mean"]) #  [0.48145466, 0.4578275, 0.40821073]
        crop_size = vision_tower.default_cfg["input_size"]
        self.crop_size = {
            'height': crop_size[1],
            'width': crop_size[2]
        }
        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(vision_tower)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def preprocess(self, img, return_tensors='pt'):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        transformed_img = self.transforms(img)
        return {'pixel_values': [transformed_img]}


class TIMMVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.load_model()

    def load_model(self):
        import os
        files = os.listdir(self.vision_tower_name)
        for file_name in files:
            if file_name.endswith('.bin'):
                bin_file = os.path.join(self.vision_tower_name, file_name)
        assert os.path.exists(bin_file)
        self.vision_tower = timm.create_model(
            self.vision_tower_name,
            pretrained=True,
            # features_only=True,
            pretrained_cfg_overlay=dict(file=bin_file),
            num_classes=0,  # remove classifier nn.Linear
            global_pool=''
        )
        self.vision_tower = self.vision_tower.eval()
        print("loaded!")

        self.image_processor = TIMMImageProcessor(self.vision_tower)
        self.is_loaded = True


    # @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0)).to(image.dtype)[:, 1:, :] # remove CLS token
                image_features.append(image_feature)
        else:
            image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype)).to(images[0].dtype)[:, 1:, :] # remove CLS token

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower._parameters[list(self.vision_tower._parameters.keys())[0]].dtype

    @property
    def device(self):
        return self.vision_tower._parameters[list(self.vision_tower._parameters.keys())[0]].device

    # @property
    # def config(self):
    #     if self.is_loaded:
    #         return self.vision_tower.config
    #     else:
    #         return self.cfg_only

    @property
    def hidden_size(self):
        return self.vision_tower.embed_dim

