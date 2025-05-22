import hydra
from src.extract_features import create_dataset


@hydra.main(config_path="../configs/", version_base=None)
def main(cfg):
    create_dataset(cfg)


if __name__ == "__main__":
    main()
