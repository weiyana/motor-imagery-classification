from utils.get_args import Args
from utils.utils import fix_random_seed, timeit
from data_loader.data_generator import DataGenerator
from models.model_builder import ModelBuilder
from trainers.trainer_maker import TrainerMaker


@timeit
def main():
    args_class = Args()
    args = args_class.args
    best_acc_per_subject=[]
    for args.subject in args.target_subject:
        args_class.preprocess()
        args_class.print_info()

        # Fix random seed
        if args.seed:
            fix_random_seed(args)

        # Load data
        data = DataGenerator(args)

        # Build model
        model = ModelBuilder(args).model

        # Make Trainer
        trainer = TrainerMaker(args, model, data).trainer

        if args.mode == 'train':
            trainer.train(best_acc_per_subject)
            # record best val acc
            best_acc_per_subject=trainer.best_acc_per_subject
        else:
            trainer.test()
        print("best_acc_per_subject:",best_acc_per_subject)


if __name__ == '__main__':
    main()
