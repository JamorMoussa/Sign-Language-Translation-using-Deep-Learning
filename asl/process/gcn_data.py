from asl.utils import HandLandMarksExtractor
from asl.datasets import AslGCNDataset
import os.path as osp
import json
import os


# SAVE_PATH = osp.join(DATASET_PATH, "raw")

# if not osp.exists(DATASET_PATH):
#     os.mkdir(DATASET_PATH)

# if not osp.exists(SAVE_PATH):
#     os.mkdir(SAVE_PATH)

# BASE_PATH = osp.join(BASE_PATH, "data")


class AslGCNDatasetProcessor:

    def __init__(
        self,
        data_dir: str,
        save_path: str,
        max_samples: int = 100
    ) -> None:
        
        self.data_dir: str = data_dir
        self.save_path: str = osp.join(save_path, "dataset")
        self.max_samples: int = max_samples

        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)
            os.mkdir(osp.join(self.save_path, "raw"))

    
    def extract_featrs_from_one_sample(
        self,
        char: str = "A",
        label: int = 0
    ) -> None:

        data = {"char": char, "label": label, "samples": []}

        char_path = osp.join(self.data_dir, char)

        for i, sample in enumerate(os.listdir(char_path)):

            if i > self.max_samples:
                break

            hand = HandLandMarksExtractor()

            try:
                
                tmp = {"id": sample.split(".")[0], "x": []}

                path = osp.join(char_path, sample)

                hand.from_image(path)

                tmp["x"] = hand.x.tolist()

                data["samples"].append(tmp)

            except:
                pass 


        json.dump(data, open(f"{osp.join(self.save_path, 'raw', char)}.json", "w"))

        print(f"{label} -> done with char: {char}", "."*10)
    


    def extract_for_all_char(self, ) -> None:

        chars = os.listdir(self.data_dir)

        for label, char in enumerate(chars):
            self.extract_featrs_from_one_sample(char, label)

    def process(self, re_process: bool = False):
        
        self.extract_for_all_char()
        dataset = AslGCNDataset(root= self.save_path, re_process= re_process)

        print("GCN dataset has been processed successfully")
        print(dataset)

        




