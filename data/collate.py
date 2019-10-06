def collate(batch):
    """ zip batches """
    return tuple(zip(*batch))
