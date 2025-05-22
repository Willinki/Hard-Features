import hydra
from src.train_mlp_on_features import train_model_on_featuresdata


@hydra.main(
    config_path="../configs/", config_name="train_on_features", version_base=None
)
def main(cfg):
    train_model_on_featuresdata(cfg)


if __name__ == "__main__":
    main()
