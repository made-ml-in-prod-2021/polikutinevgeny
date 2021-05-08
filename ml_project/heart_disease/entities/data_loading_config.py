from dataclasses import dataclass, field

import omegaconf

from heart_disease.entities.splitting_config import SplittingConfig


@dataclass
class DataLoadingConfig:
    split_config: SplittingConfig = field(default_factory=lambda: SplittingConfig)
    data_path: str = omegaconf.MISSING
