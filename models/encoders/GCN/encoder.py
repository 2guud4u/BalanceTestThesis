import torch
import torch.nn as nn
import torch.nn.functional as F

# Skeleton edge definitions for different formats
MEDIAPIPE_EDGES = [
    # Arms (reindexed) and torso/legs connections only — no face/mouth edges
    (0, 1), (1, 3), (3, 5), (5, 7), (7, 9), (5, 9),      # Right arm path (was 11..20)
    (0, 2), (2, 4), (4, 6), (6, 8), (4, 8),              # Left arm path (was 11..19 variants)
    (1, 13), (0, 12), (12, 13),                          # Torso connections (was 12,23,24)
    (12, 14), (14, 16), (16, 18), (18, 20), (16, 20),    # Left leg chain (reindexed)
    (13, 15), (15, 17), (17, 19), (19, 21), (17, 21)     # Right leg chain (reindexed)
]

MOTIONBERT_EDGES = [
 (0, 1), (1, 2), (2, 3),        # Right leg
    (0, 4), (4, 5), (5, 6),        # Left leg
    (0, 7), (7, 8), (8, 9), (9,10),# Spine → head
    (8,11), (11,12), (12,13),      # Left arm
    (8,14), (14,15), (15,16)       # Right arm
]

NTU_EDGES = [
    # NTU skeleton (25 joints)
    # https://github.com/shahroudy/NTURGB-D
    # Spine chain
    (0, 1), (1, 20), (20, 2), (2, 3),
    # Left arm
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 22),
    # Right arm
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23),
    # Left leg
    (0, 12), (12, 13), (13, 14), (14, 15),
    # Right leg
    (0, 16), (16, 17), (17, 18), (18, 19),
    # Head
    (3, 21), (3, 24)
]

def get_edges_for_num_joints(num_joints):
    """
    Return appropriate edge list based on number of joints.
    """
    if num_joints == 33:
        return MEDIAPIPE_EDGES
    elif num_joints == 17:
        return MOTIONBERT_EDGES
    elif num_joints == 25:
        return NTU_EDGES
    else:
        # Default: create a simple chain for unknown formats
        print(f"Warning: Unknown skeleton format with {num_joints} joints. Using simple chain topology.")
        return [(i, i+1) for i in range(num_joints-1)]
def build_adjacency(num_nodes, edges, self_loops=True):
    A = torch.zeros(num_nodes, num_nodes)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    if self_loops:
        A.fill_diagonal_(1)
    return A

class SimpleGraphConv(nn.Module):
    """
    Graph convolution using:
      A = A_fixed + A_learned (optional)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        num_nodes,
        edges,
        use_adaptive=True
    ):
        super().__init__()

        self.in_ch = in_channels
        self.out_ch = out_channels
        self.num_nodes = num_nodes

        # linear transform on node features
        self.conv = nn.Linear(in_channels, out_channels, bias=False)

        # fixed adjacency from skeleton
        A_fixed = build_adjacency(num_nodes, edges)
        self.register_buffer("A_fixed", A_fixed)

        # optional learned residual adjacency
        if use_adaptive:
            self.A_learned = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        else:
            self.A_learned = None

        self.bn = nn.BatchNorm1d(num_nodes * out_channels)

    def forward(self, x):
        """
        x: (B, T, J * in_ch)
        """
        B, T, Fdim = x.shape
        J = self.num_nodes
        assert Fdim == J * self.in_ch

        # reshape → (B*T, J, in_ch)
        x = x.reshape(B * T, J, self.in_ch)

        # linear projection per joint
        xw = self.conv(x)  # (B*T, J, out_ch)

        # adjacency
        if self.A_learned is not None:
            A = self.A_fixed + self.A_learned
        else:
            A = self.A_fixed

        A = F.softmax(A, dim=-1)

        # graph propagation
        out = torch.matmul(A, xw)  # (B*T, J, out_ch)

        # reshape + norm
        out = out.reshape(B * T, J * self.out_ch)
        out = self.bn(out)
        out = F.relu(out)
        out = out.reshape(B, T, J * self.out_ch)

        return out
    
class SpatialEncoder(nn.Module):
    def __init__(self, num_joints=33, in_channels=3, edges=None, output_dim=128):
        """
        Dynamic spatial encoder that supports different skeleton formats.
        
        Args:
            num_joints: Number of joints in skeleton (default: 33 for MediaPipe)
            in_channels: Input channels per joint (default: 3 for x,y,z)
            edges: List of (i, j) edge tuples for graph. If None, auto-detects based on num_joints
                   Supports: 33 (MediaPipe), 17 (MotionBert), 25 (NTU), or custom
            output_dim: Output feature dimension (default: 128)
        """
        super().__init__()
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.out_dim = output_dim
        
        if edges is None:
            edges = get_edges_for_num_joints(num_joints)
        
        self.g1 = SimpleGraphConv(
            in_channels=in_channels,
            out_channels=64,
            num_nodes=num_joints,
            edges=edges,
            use_adaptive=True
        )
        self.g2 = SimpleGraphConv(
            in_channels=64,
            out_channels=128,
            num_nodes=num_joints,
            edges=edges,
            use_adaptive=True
        )
        # Projection: (J*128) -> output_dim
        self.proj = nn.Linear(num_joints * 128, output_dim)
        
    def forward(self, x):
        """
        x: (B, T, J*C) flattened skeleton features
        Returns: (B, out_dim, T) features for MS-TCN2 compatibility
        """
        x = self.g1(x)
        x = self.g2(x)
        B, T, F = x.shape
        x = x.view(B * T, F)
        x = self.proj(x)
        x = x.view(B, T, -1)  # (B, T, out_dim)
        x = x.permute(0, 2, 1).contiguous()  # (B, out_dim, T)
        return x


