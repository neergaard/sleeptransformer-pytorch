import torch


def collate(batch):

        batch_data = torch.stack([torch.Tensor(el["signal"]) for el in batch])
        batch_stages = torch.stack([torch.LongTensor(el["stages"]) for el in batch])
        batch_records = [el["record"] for el in batch]

        return {'data': batch_data, 'targets': batch_stages, 'records': batch_records}
