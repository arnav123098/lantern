import torch

'''
- saves training state
- loads state to resume training

TODO: testing
'''
class Checkpoint:
    @staticmethod
    def save(
        path: str,
        step: int,
        model,
        optimizer = None,
        dataloader = None,
        extra: dict | None = None
    ):
        checkpoint = {
            'step': step,
            'model': model.state_dict(),
            'meta': {
                'lantern_version': '0.1.0',
                'model_name': model.__class__.__name__
            }
        }

        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()

        if dataloader is not None:
            checkpoint['dataloader'] = dataloader.state_dict()

        if extra is not None:
            checkpoint['extra'] = extra.state_dict()

        torch.save(checkpoint, path)

    @staticmethod
    def load(
        path: str,
        model,
        optimizer = None,
        dataloader = None
    ):
        
        checkpoint = torch.load(path, map_location='cpu')

        model.load_state_dict(checkpoint['model'])

        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        if dataloader is not None and 'dataloader' in checkpoint:
            dataloader.load_state_dict(checkpoint['dataloader'])

        return checkpoint
