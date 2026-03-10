class Qwen3VisionConfig:
    def __init__(
        self,
        hidden_size=1024,  # Qwen3-VL-2B 默认值（官方值）
        intermediate_size=4096,  # 官方值
        depth=24,  # Qwen3-VL-2B 的层数（官方值）
        num_attention_heads=16,  # 官方值
        num_channels=3,
        in_channels=3,
        patch_size=16,  # Qwen3-VL-2B 的 patch_size（官方值）
        temporal_patch_size=2,  # 官方值
        spatial_merge_size=2,  # 官方值
        num_position_embeddings=2304,  # 官方值
        out_hidden_size=2048,  # 官方值
        hidden_act="gelu_pytorch_tanh",  # 官方值
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens=None,
        deepstack_visual_indexes=[5, 11, 17],  # DeepStack技术（官方值）
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.depth = depth
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.num_position_embeddings = num_position_embeddings
        self.out_hidden_size = out_hidden_size
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens
        self.deepstack_visual_indexes = deepstack_visual_indexes


class Qwen3TextConfig:
    def __init__(
        self,
        vocab_size,
        hidden_size=2048,  # Qwen3-VL-2B 默认值（官方值）
        intermediate_size=6144,  # 官方值
        num_hidden_layers=28,  # Qwen3-VL-2B 层数（官方值）
        num_attention_heads=16,  # 官方值
        num_key_value_heads=8,  # 官方值
        head_dim=128,  # 官方值
        max_position_embeddings=262144,  # 官方值
        rms_norm_eps=1e-6,
        rope_theta=5000000,  # 官方值
        attention_bias=False,  # 官方值
        attention_dropout=0.0,
        pad_token_id=None,
        hidden_act="silu",  # 官方值
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id
        self.hidden_act = hidden_act


class Qwen3VLConfig:
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=151655,  # Qwen3-VL-2B 的图像 token ID（官方值）
        video_token_id=151656,  # 官方值
        vision_start_token_id=151652,  # 官方值
        vision_end_token_id=151653,  # 官方值
        vocab_size=151936,  # Qwen3-VL-2B 的 vocab size（官方值）
        projection_dim=2048,  # 官方值（out_hidden_size）
        hidden_size=2048,  # 官方值
        pad_token_id=None,
        bos_token_id=151643,  # 官方值
        eos_token_id=151645,  # 官方值
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # 使用官方配置类
        self.vision_config = Qwen3VisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = Qwen3TextConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        # 计算图像token数量
        # self.text_config.num_image_tokens = (
        #     self.vision_config.num_position_embeddings // self.vision_config.spatial_merge_size
        # )
        self.vision_config.projection_dim = projection_dim