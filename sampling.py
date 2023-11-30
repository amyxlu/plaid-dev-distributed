from k_diffusion.callback import main
from k_diffusion.config import SampleCallbackConfig


if __name__ == "__main__":
    import tyro
    args = tyro.cli(SampleCallbackConfig)
    main(args)
