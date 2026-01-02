# if __name__ == "__main__":
#     from config.settings import config
#     from config.config_groups import Environment, GameType, ModelName, TargetClass, InferenceConfig, DetectionResult
# else:
#     from .settings import config #init itself can run only when running without . in configManager
#     from .config_groups import Environment, GameType, ModelName, TargetClass, InferenceConfig, DetectionResult

# __all__ = ['Environment', 'GameType']
from .settings import *
from .constants_groups import *
from .config_groups import *