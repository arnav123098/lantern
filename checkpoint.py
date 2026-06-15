import torch

'''
- saves training state
- loads state to resume training
'''
class Checkpoint:
    @staticmethod
    def save(
        path: str,
        step: int,
        model,
        optimizers: list | None = None,
        dataloaders: list | None = None,
        trainer = None,
        extra: dict | None = None
    ):
        checkpoint = {
            'step': step,
            'model': model.state_dict(),
            'optimizers': [],
            'dataloaders': [],
            'trainer': None,
            'extra': None,
            'meta': {
                'lantern_version': '0.1.0',
                'model_name': model.__class__.__name__
            }
        }

        if optimizers is not None:
            for optimizer in optimizers:
                checkpoint['optimizers'].append(optimizer.state_dict())

        if dataloaders is not None:
            for dataloader in dataloaders:
                checkpoint['dataloaders'].append(dataloader.state_dict())
        
        if trainer is not None:
            checkpoint['trainer'] = trainer.state_dict()

        if extra is not None:
            checkpoint['extra'] = extra

        torch.save(checkpoint, path)

    @staticmethod
    def load(
        path: str,
        model,
        optimizers: list | None = None,
        dataloaders: list | None = None,
        trainer = None,
        device: str = "cpu"
    ):
        
        checkpoint = torch.load(path, map_location=device)
        
        model.load_state_dict(checkpoint['model'])

        if optimizers is not None and 'optimizers' in checkpoint:
            assert len(optimizers) == len(checkpoint['optimizers'])

            for i, optimizer in enumerate(optimizers):
                optimizer.load_state_dict(checkpoint['optimizers'][i])

        if dataloaders is not None and 'dataloaders' in checkpoint:
            assert len(dataloaders) == len(checkpoint['dataloaders'])

            for i, dataloader in enumerate(dataloaders):
                dataloader.load_state_dict(checkpoint['dataloaders'][i])

        if trainer is not None and 'trainer' in checkpoint:
            trainer.load_state_dict(checkpoint['trainer'])

        return checkpoint
