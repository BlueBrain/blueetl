from common.utils import L, run_analysis


@run_analysis
def main(analysis_config: dict) -> dict:
    L.info("analysis_config:\n%s", analysis_config)
    return {"outputs": []}
