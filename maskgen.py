import torch
import math
from torch import nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.smoother import smoother, exponential_smoother
from txai.utils.functional import transform_to_attn_mask
from txai.models.encoders.positional_enc import PositionalEncodingTF
from tint.models import MLP, RNN

trans_decoder_default_args = {
    "nhead": 1, 
    "dim_feedforward": 32, 
}

MAX = 10000.0

import torch
from torch import Tensor
import warnings


def decoupled_gumbel_softmax(
        logits: Tensor,
        tau_forward: float = 1.0,
        tau_backward: float = 1.0,
        adapt_lr: float = 0.1,
        eps: float = 1e-10,
        dim: int = -1,
        adapt_tau: bool = True,
        tau_min: float = 0.1,
        tau_max: float = 10.0,

) -> Tensor:
    """
    Sample from the Decoupled Gumbel-Softmax distribution with separate forward and backward temperatures.
    The forward pass uses tau_forward for prediction, and the backward pass uses tau_backward for gradient computation.

    Args:
        logits: `[..., num_features]` unnormalized log probabilities.
        tau_forward: initial forward temperature for prediction.
        tau_backward: initial backward temperature for gradient computation.
        hard: if ``True``, the returned samples will be discretized as one-hot vectors,
              but will be differentiated as if it is the soft sample in autograd.
        dim: A dimension along which softmax will be computed. Default: -1.
        adapt_tau: if ``True``, adaptively adjust temperatures during training.
        tau_min: minimum allowable temperature during adaptation.
        tau_max: maximum allowable temperature during adaptation.
        adapt_lr: learning rate for temperature adjustment.

    Returns:
        Sampled tensor of same shape as `logits` from the Decoupled Gumbel-Softmax distribution.
        If ``hard=True``, the returned samples will be one-hot, otherwise they will
        be probability distributions that sum to 1 across `dim`.

    Notes:
        This function implements the Decoupled Straight-Through Gumbel-Softmax method, allowing for independent
        control of forward and backward relaxation smoothness, with optional adaptive temperature adjustment.
    """

    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    # Initialize temperature gates
    tau_forward_gate = tau_forward
    tau_backward_gate = tau_backward

    # Adaptive adjustment logic (example heuristic based on logits variance)
    logits_std = torch.flatten(logits).std(dim=dim, unbiased=False)
    tau_forward_gate = tau_forward - adapt_lr * logits_std.clamp(min=0, max=1000)
    tau_backward_gate = tau_backward + adapt_lr * logits_std.clamp(min=0, max=1000)

    # Ensure temperatures stay within bounds
    tau_forward_gate = tau_forward_gate.clamp(min=tau_min, max=tau_max)
    tau_backward_gate = tau_backward_gate.clamp(min=tau_min, max=tau_max)

    # Forward pass with tau_forward_gate for prediction
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels_forward = (logits + gumbels) / tau_forward_gate  # Forward Gumbel(logits, tau_forward_gate)
    y_soft_forward = gumbels_forward.softmax(dim)

    # Straight-through for forward pass
    index = y_soft_forward.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    y_forward = y_hard - y_soft_forward.detach() + y_soft_forward

    # # Backward pass with tau_backward_gate for gradient computation
    # gumbels_backward = (logits + gumbels) / tau_backward_gate  # Backward Gumbel(logits, tau_backward_gate)
    # y_soft_backward = gumbels_backward.softmax(dim)

    # Use y_soft_backward in gradient computation
    if gumbels_forward.requires_grad:
        gumbels_forward.register_hook(lambda grad: grad * (tau_forward_gate / tau_backward_gate))

    return y_forward


class MaskGenerator(nn.Module):
    def __init__(self, 
            d_z, 
            max_len,
            d_pe = 16,
            trend_smoother = False,
            agg = 'max',
            pre_agg_mlp_d_z = 32,
            time_net_d_z = 64,
            trans_dec_args = trans_decoder_default_args,
            n_dec_layers = 2,
            tau = 1.0,
            use_ste = True,

            use_decoupled_ste=False,
            forward_ste_temperature=0.1,
            backward_ste_temperature=3.0,
            ste_learning_rate=0.001,
        ):
        super(MaskGenerator, self).__init__()

        self.d_z = d_z
        self.pre_agg_mlp_d_z = pre_agg_mlp_d_z
        self.time_net_d_z = time_net_d_z
        self.agg = agg
        self.max_len = max_len
        self.trend_smoother = trend_smoother
        self.use_ste = use_ste
        self.d_inp = self.d_z - d_pe
        self.tau = tau

        self.use_decoupled_ste = use_decoupled_ste
        self.forward_ste_temperature = forward_ste_temperature
        self.backward_ste_temperature = backward_ste_temperature
        self.ste_learning_rate = ste_learning_rate

        dec_layer = nn.TransformerDecoderLayer(d_model = d_z, **trans_dec_args) 
        self.mask_decoder = nn.TransformerDecoder(dec_layer, num_layers = n_dec_layers)

        # self.mask_decoder = nn.Sequential(
        #         RNN(
        #             input_size=d_z,
        #             rnn="gru",
        #             hidden_size=d_z,
        #             bidirectional=True,
        #         ),
        #         MLP([2 * d_z, d_z]),
        #     )

        
        self.pre_agg_net = nn.Sequential(
            nn.Linear(d_z, self.pre_agg_mlp_d_z),
            nn.PReLU(),
            nn.Linear(self.pre_agg_mlp_d_z, self.pre_agg_mlp_d_z),
            nn.PReLU(),
        )

        if self.d_inp > 1:
            self.time_prob_net = nn.Sequential(nn.Linear(d_z, self.d_inp), nn.Sigmoid())
        else:
            self.time_prob_net = nn.Linear(d_z, 2)

        # if self.d_inp > 1:
        #     self.time_prob_net =  MLP([d_z, 64, self.d_inp])
        #     self.time_prob_net_std = MLP([d_z, 64, self.d_inp])

        # else:
        #     self.time_prob_net =  MLP([d_z, 64, 2])
        #     self.time_prob_net_std = MLP([d_z, 64, 2])


        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        self.init_weights()

    def init_weights(self):
        def iweights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.time_prob_net.apply(iweights)
        self.pre_agg_net.apply(iweights)

    def reparameterize(self, total_mask):

        if self.d_inp == 1:
            if total_mask.shape[-1] == 1:
                # Need to add extra dim:
                inv_probs = 1 - total_mask
                total_mask_prob = torch.cat([inv_probs, total_mask], dim=-1)
            else:
                total_mask_prob = total_mask.softmax(dim=-1)
        else:
            # Need to add extra dim:
            inv_probs = 1 - total_mask
            total_mask_prob = torch.stack([inv_probs, total_mask], dim=-1)

        if self.use_decoupled_ste:
            total_mask_reparameterize = decoupled_gumbel_softmax(
                                            torch.log(total_mask_prob + 1e-9),
                                            tau_forward = self.forward_ste_temperature,
                                            tau_backward=self.backward_ste_temperature,
                                            adapt_lr = self.ste_learning_rate,
                                            )[...,1]
        else:
            total_mask_reparameterize = F.gumbel_softmax(torch.log(total_mask_prob + 1e-9), tau=self.tau, hard=self.use_ste)[..., 1]

        return total_mask_reparameterize, total_mask_prob

    def forward(self, z_seq, src, times, get_agg_z = False):

        x = torch.cat([self.pos_encoder(times), src], dim=-1)  # t bs n
        # x = torch.cat([src, self.pos_encoder(times)], dim = -1) # t bs n

        # x = x.transpose(1,0) ###########


        if torch.any(times < -1e5):
            tgt_mask = (times < -1e5).transpose(0,1)
        else:
            tgt_mask = None

        z_seq_dec = self.mask_decoder(tgt = x, memory = z_seq, tgt_key_padding_mask = tgt_mask)

        # z_seq_dec = self.mask_decoder(x)  ###########
        # z_seq_dec = z_seq_dec.transpose(1,0)  ###########
        
        z_pre_agg = self.pre_agg_net(z_seq_dec)

        # mean = self.time_prob_net(z_seq_dec)
        # std = self.time_prob_net_std(z_seq_dec)
        # p_time = self.gauss_sample(mean, std)
        p_time = self.time_prob_net(z_seq_dec)

        total_mask_reparameterize, total_mask_prob = self.reparameterize(p_time.transpose(0,1))
        if self.d_inp == 1:
            total_mask = p_time.transpose(0,1).softmax(dim=-1)[...,1].unsqueeze(-1)
        else:
            total_mask = p_time.transpose(0,1) # Already sigmoid transformed

        # Transpose both src and times below bc expecting batch-first input

        # TODO: Get time and cycle returns later

        if get_agg_z:
            agg_z = z_pre_agg.max(dim=0)[0]
            return total_mask, total_mask_reparameterize, agg_z
        else:
            return total_mask, total_mask_reparameterize
        
    def gauss_sample(self, mean_logit, std, training=True):
        if training:
            att_bern = (mean_logit + std * torch.randn(mean_logit.shape, device=mean_logit.device)).sigmoid()
        else:
            att_bern = (mean_logit).sigmoid()
        return att_bern