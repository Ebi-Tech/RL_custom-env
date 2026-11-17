"""Beyond Sale Shop environment package: registers BeyondSaleShop-v0 and exports ShopEnv."""
import gymnasium

from environment.custom_env import ACTION_NAMES, ShopEnv

ENV_ID = "BeyondSaleShop-v0"

if ENV_ID not in gymnasium.registry:
    gymnasium.register(id=ENV_ID, entry_point="environment.custom_env:ShopEnv")

__all__ = ["ACTION_NAMES", "ENV_ID", "ShopEnv"]
