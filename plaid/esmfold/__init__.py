ESMFOLD_S_DIM = 1024  # dimension of the s_s_0 tensor input to ESMFold folding trunk
ESMFOLD_Z_DIM = 128   # dimension of the paired representation s_z_0 input
from .esmfold import ESMFold, ESMFoldConfig, get_esmfold_model_state
from .tri_self_attn_block import TriangularSelfAttentionBlock
from .trunk import RelativePosition, FoldingTrunk, FoldingTrunkConfig