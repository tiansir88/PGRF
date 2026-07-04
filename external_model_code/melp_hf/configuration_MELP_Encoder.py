from transformers import PretrainedConfig


class MELPEncoderConfig(PretrainedConfig):
    model_type = "melp"

    def __init__(
        self,
        model_size: str = "small", # small by default
        shared_emb_dim: int = 256,
        embed_dim_caption: int = 768,
        use_attentional_pool_contrast: bool = True,
        use_attentional_pool_caption: bool = True,
        n_queries_contrast: int = 14,
        n_queries_caption: int = 128,
        attn_pooler_heads: int = 8,
        proj: str = "linear",
        drop: float = 0.,
        proj_bias: bool = False,
        num_leads: int = 12,
        **kwargs
    ):
        self.model_size = model_size
        self.shared_emb_dim = shared_emb_dim
        self.embed_dim_caption = embed_dim_caption
        self.use_attentional_pool_contrast = use_attentional_pool_contrast
        self.use_attentional_pool_caption = use_attentional_pool_caption
        self.n_queries_contrast = n_queries_contrast
        self.n_queries_caption = n_queries_caption
        self.attn_pooler_heads = attn_pooler_heads
        self.proj = proj
        self.drop = drop
        self.proj_bias = proj_bias
        self.num_leads = num_leads
        super().__init__(**kwargs)


