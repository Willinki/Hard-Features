import hydra
from src.train_model import train_model


@hydra.main(config_path="../configs/", version_base=None)
def main(cfg):
    train_model(cfg)


if __name__ == "__main__":
    main()
