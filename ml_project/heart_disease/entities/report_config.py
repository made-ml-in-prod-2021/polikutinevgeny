from dataclasses import dataclass


@dataclass
class ReportConfig:
    # Right now hydra and OmegaConf don't support Path type :(
    data_path: str
    report_path: str
