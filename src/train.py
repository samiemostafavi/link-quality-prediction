from easy_tpp.config_factory import Config
from src.tpprunner import TPPRunner

def train_model(args):

    config = Config.build_from_yaml_file(args.config, experiment_id=args.id)
    model_runner = TPPRunner(config)
    model_runner.run()
