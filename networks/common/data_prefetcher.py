import torch

from networks.common.io_tools import dict_to_non_block

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()


    def preload(self):
        try:
            self.next_data, _ = next(self.loader)
        except StopIteration:
            self.next_data = None

            return
        with torch.cuda.stream(self.stream):
            self.next_data = dict_to_non_block(self.next_data, device)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data

        self.preload()
        return data
