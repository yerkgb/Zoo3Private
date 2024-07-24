from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
import torch as th
import torch.nn as nn

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                                           activation_fn=nn.Tanh)
        self.mlp_extractor = MlpExtractor(self.features_dim, net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                                          activation_fn=nn.Tanh)
    
    def _build_mlp_extractor(self):
        self.mlp_extractor = MlpExtractor(self.features_dim, net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                                          activation_fn=nn.Tanh)

# Custom policy network with entropy regularization
policy_kwargs = {
    'features_extractor_class': CustomPolicy,
    'features_extractor_kwargs': dict(features_dim=64),
    'optimizer_class': th.optim.Adam,
    'optimizer_kwargs': dict(lr=2.5e-4)
}
