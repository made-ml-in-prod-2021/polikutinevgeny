import logging
from pathlib import Path

import hydra
import pandas as pd
from pandas_profiling import ProfileReport

from heart_disease.data.make_dataset import read_data
from heart_disease.entities.report_config import ReportConfig

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_report(data: pd.DataFrame) -> ProfileReport:
    return ProfileReport(
        df=data,
        title="Heart disease dataset",
        explorative=True,
        progress_bar=False,
        lazy=False,
    )


def save_report(report: ProfileReport, path: str):
    report.to_file(path)


@hydra.main(config_path=PROJECT_ROOT / "config", config_name="report_config")
def make_report(cfg: ReportConfig):
    log.info("Reading data file '%s'", cfg.data_path)
    data = read_data(cfg.data_path)
    log.info("Building report")
    report = build_report(data)
    log.info("Saving report to '%s", cfg.report_path)
    save_report(report, cfg.report_path)


if __name__ == '__main__':
    make_report()
