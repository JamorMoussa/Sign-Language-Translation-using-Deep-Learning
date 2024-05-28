from torch_geometric.data import Data, InMemoryDataset
import os.path as osp
import torch
import json
import os


class DatasetNotExist(Exception):
    def __init__(self):
        super(DatasetNotExist, self).__init__("The 'dataset' doesn't exist, run The data preprocessing first.")


class AslGCNDataset(InMemoryDataset):

    def __init__(
            self,
            root: str,
            re_process: bool = False,
            transform=None,
            pre_transform=None,
            pre_filter=None
        ) -> None:
        
        if not (osp.exists(root) and osp.exists(osp.join(root, "raw"))):
            raise DatasetNotExist()
        
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=re_process)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return sorted(os.listdir(self.raw_dir))

    @property
    def processed_file_names(self):
        return ['data.pt']

    def edge_index(self) -> torch.tensor:
        return torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 5, 6, 6, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 12, 13, 13, 13, 14, 14, 15, 15, 16, 17, 17, 17, 18, 18, 19, 19, 20],
                             [1, 5, 17, 0, 2, 1, 3, 2, 4, 3, 0, 6, 9, 5, 7, 6, 8, 7, 5, 10, 13, 9, 11, 10, 12, 11, 9, 14, 17, 13, 15, 14, 16, 15, 0, 13, 18, 17, 19, 18, 20, 19]]
                            ).type(torch.long)

    def download(self):
        ...

    def process(self):

        data_list = []

        chars = []

        for char in self.raw_file_names:
            
            jsn = json.load(open(osp.join(self.raw_dir, char), "r"))

            y = torch.tensor(jsn["label"])

            if len(samples := jsn["samples"]) != 0:

                chars.append(char)

                for sample in samples:

                    x = torch.tensor(sample["x"]).type(torch.float32)

                    data = Data(edge_index=self.edge_index(), x=x, y=y)

                    data_list.append(data)


            self.data, self.slices = self.collate(data_list)

            torch.save((self._data, self.slices), self.processed_paths[0])


        