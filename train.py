from k_diffusion import Trainer
from k_diffusion.config import TrainArgs, dataclass_to_dict

def main(args: TrainArgs):
    trainer = Trainer(args)
    trainer.run()


if __name__ == "__main__":
    import tyro
    from pprint import pprint

    # Parse config with overrides from command line; otherwise uses defaults.
    args = tyro.cli(TrainArgs)
    pprint(dataclass_to_dict(args))

    if args.debug_mode:
        try:
            main(args)
        except:
            import pdb, sys, traceback

            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
    else:
        main(args)