from pathlib import Path

from sitstart import USER_DATA_ROOT

CONFIG_ROOT = str(Path(__file__).resolve().parent / "conf")
HYDRA_VERSION_BASE = "1.2"
RUN_ROOT = f"{USER_DATA_ROOT}/{__name__.replace('.', '/')}/runs"
TEST_ROOT = f"{USER_DATA_ROOT}/{__name__.replace('.', '/')}/tests"
TRIAL_ARCHIVE_URL = "s3://sitstart/archive/trials"
