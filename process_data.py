from asl.process import AslGCNDatasetProcessor
import argparse




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract Land Marks features from ASL dataset.')
    
    parser.add_argument('--which', type= str, default= "gcn", 
                        help= "dataset you want to process. Svailable choices: 'gcn', 'cnn' and 'mlp'")

    parser.add_argument('--max_samples', type=int, default=10,
                        help='maximum number of samples to process for each character')
    
    parser.add_argument('--replace', type=bool, default=False,
                        help='Boolean determines, if the process.py replace the old dataset')

    args = parser.parse_args()

    if args.which.lower() == "gcn":

        processor = AslGCNDatasetProcessor(
            data_dir= "./data/asldataset",
            save_path= "./data",
            max_samples= args.max_samples
        )

        processor.process(re_process= args.replace)