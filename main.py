import argparse
from src.preprocess import preprocess_edaf, plot_link_data, create_training_dataset
from src.train import train_model
from src.predict import generate_predictions, plot_predictions
from src.evaluate import evaluate_model

# python main.py -t preprocess -s data/240928_082545_results
# python main.py -t plot_link_data -s data/240928_082545_results -c config/dataset_config.json -n test0
# python main.py -t create_training_dataset -s data/240928_082545_results -c config/dataset_config.json -n test0
# python main.py -t train_model -c config/training_config.yaml -i THP_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link Quality Prediction")
    parser.add_argument("-t", "--task", choices=[
            "preprocess", 
            "plot_link_data", 
            "create_training_dataset",
            "train_model",
            "generate_predictions",
            "plot_predictions",
            "evaluate",
        ], 
        help="Specify the task to run"
    )
    parser.add_argument("-p", "--predict", choices=[
                "probabilistic",
                "sampling",
        ],
        help="Specify the prediction method")
    parser.add_argument("-s", "--source", help="Specify the source directory")
    parser.add_argument("-f", "--fast", action="store_true", help="Specify if in plot_link_data, only priliminary data should be plotted")
    parser.add_argument("-c", "--config", help="Specify the configuration file")
    parser.add_argument("-g", "--configname", help="Specify the configuration name in the configuration file")
    parser.add_argument("-n", "--name", help="Specify the name of the dataset")
    parser.add_argument("-i", "--id", help="Specify the training id")
    args = parser.parse_args()

    if args.task == "preprocess":
        preprocess_edaf(args)
    elif args.task == "plot_link_data":
        plot_link_data(args)
    elif args.task == "create_training_dataset":
        create_training_dataset(args)
    elif args.task == "train_model":
        train_model(args)
    elif args.task == "generate_predictions":
        generate_predictions(args)
    elif args.task == "plot_predictions":
        plot_predictions(args)
    elif args.task == "evaluate":
        evaluate_model(args)
    else:
        print("Invalid task specified")

        

        

