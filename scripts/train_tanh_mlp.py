import hydra
from src.train_mlp import train_model


@hydra.main(config_path="../configs/", config_name="train_tanh_mlp", version_base=None)
def main(cfg):
    train_model(cfg)


if __name__ == "__main__":
    main()
