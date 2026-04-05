from shinka.utils.utils_hydra import build_cfgs_from_python
from shinka.core import EvolutionRunner
from dotenv import load_dotenv
import os

load_dotenv()

assert os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY is not set"
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is not set"


# command line configs can be overridden both with lists of arguments or a
# Python dictionary of keyword arguments that will be converted to the
# appropriate arguments

launcher_args = [
    "variant=novelty_generator_example",
    "database=island_small",
    "evolution=small_budget_vida",
    "evo_config.num_generations=10",
]

launcher_kwargs = {
    "evo_config.llm_models": ["gemini-3-flash-preview"],
    "evaluate_function.llm_judge_names": ["gemini-3-pro-preview"],
}


job_cfg, db_cfg, evo_cfg, cfg = build_cfgs_from_python(
    *launcher_args, **launcher_kwargs
)

evo_runner = EvolutionRunner(
    evo_config=evo_cfg,
    job_config=job_cfg,
    db_config=db_cfg,
    verbose=cfg.verbose,
)

evo_runner.run()
