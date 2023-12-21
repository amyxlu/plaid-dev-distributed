from plaid.callbacks import main
from plaid.config import SampleCallbackConfig


if __name__ == "__main__":
    import tyro
    args = tyro.cli(SampleCallbackConfig)
    main(args)
