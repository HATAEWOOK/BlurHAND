from base import Trainer

if __name__ == "__main__":
    trainer = Trainer()
    trainer._make_batch_loader()
    print(len(trainer.train_loader))
    iter_loader = iter(trainer.train_loader)
    sample = next(iter_loader)