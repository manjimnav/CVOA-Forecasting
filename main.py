from CVOA.CVOA import CVOA
from ETL.ETL import *
from ETL.datagenerator import DataGenerator
from DEEP_LEARNING.LSTM import *
import time as time
import argparse
from DEEP_LEARNING.LSTM import fit_model, getMetrics_denormalized
import tensorflow as tf

try:
    from gooey import Gooey

    use_gooey = True
except:
    use_gooey = False
    Gooey = None
    print("Gooey not enabled")


class Fitness():
    def __init__(self, prediction_horizon, scaler, xtrain=None, ytrain=None, xval=None, yval=None, train_gen=None, val_gen=None, epochs=10, batch=1024, model='lstm', window=None, natts=None):
        self.train_generator = train_gen
        self.model = model
        self.window = window
        self.natts = natts
        self.pred_horizon = prediction_horizon
        self.batch = batch
        self.epochs = epochs
        self.scaler = scaler
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xval = xval
        self.yval = yval
        self.val_gen = val_gen

    def __call__(self, individual):
        mse, mape, mae = fit_model(train_gen=self.train_generator, val_gen=self.val_gen,
                                          xtrain=self.xtrain, ytrain=self.ytrain, xval=self.xval, yval=self.yval,
                                          individual_fixed_part=individual.fixed_part,
                                          individual_variable_part=individual.var_part, scaler=self.scaler,
                                          prediction_horizon=self.pred_horizon, epochs=self.epochs, batch=self.batch,
                                          model=self.model, window=self.window, natts=self.natts)
        print(individual)
        print("---\n" + "MSE: ", " {:.4f}".format(mse[0]) + " ; MAPE: ", " {:.4f}".format(mape[0]) + " ; MAE: ",
              " {:.4f}".format(mae[0]) + "\n---")
        individual.fitness = mape[0]
        return individual  # , model


def use_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')  # Obtener la lista de GPU's instaladas en la maquina
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def GooeyDec(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)

    return decorator


@GooeyDec(Gooey, use_gooey)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Path to the data", required=True)
    parser.add_argument("-g", "--generator", help="Use generator in the data", action="store_true", default=False)
    parser.add_argument("-w", "--window", type=int, help="Window size", default=168)
    parser.add_argument("-ho", "--horizon", type=int, help="Horizon size", default=24)
    parser.add_argument("-t", "--test", help="Test size", default=.3)
    parser.add_argument("-v", "--val", help="Validation size", default=.3)
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=10)
    parser.add_argument("-b", "--batch", type=int, help="Batch size", default=1024)
    parser.add_argument("-dc", "--datecolumn", help="Date column name", default='FECHA_HORA')
    parser.add_argument("-he", "--header", help="Header row", type=int, default=None)
    parser.add_argument("-n", "--normalize", help="Normalize data", action="store_true", default=True)
    parser.add_argument("-tg", "--target", help="Target attribute", default=None)
    parser.add_argument("-m", "--model", help="Type of model (encoder-decoder,  encoder-attention-decoder, lstm)",
                        choices=['encoder-decoder', 'encoder-attention-decoder', 'lstm'], default='lstm')
    parser.add_argument("-p", "--processes", help="Number of processes (Default 2)",
                        type=int, default=2)
    parser.add_argument("-gpu", "--use_gpu", help="Use GPU in training",
                        action="store_true", default=False)
    parser.add_argument("-uv", "--use_variable", help="Use variable size in optimization",
                        action="store_true", default=True)
    parser.add_argument("-di", "--discrete", help="Discrete optimization problem",
                        action="store_true", default=True)

    args = parser.parse_args()
    # Deep Learning parameters
    epochs = args.epochs
    batch = args.batch

    if args.use_gpu:
        use_gpu()

    # Load the dataset
    data, scaler, scaler_target = load_data(path_to_data=args.data, useNormalization=True, date_column=args.datecolumn,
                                            use_generator=args.generator, header=args.header, target=args.target)
    if not args.generator:
        # Transform data to a supervised dataset
        data = data_to_supervised(data, historical_window=args.window, prediction_horizon=args.horizon)

    # Split the dataset
    xtrain, xtest, ytrain, ytest, xval, yval = splitData(data, historical_window=args.window, test_size=args.test,
                                                         val_size=args.val, use_generator=args.generator,
                                                         target=args.target)

    if not args.generator:
        # Add shape to use LSTM network
        xtrain, xtest, ytrain, ytest, xval, yval = adaptShapesToLSTM(xtrain, xtest, ytrain, ytest, xval, yval)

        fitness = Fitness(xtrain=xtrain, ytrain=ytrain, xval=ytest, yval=yval,
                          prediction_horizon=24, epochs=epochs, batch=batch, scaler=scaler_target,
                          model=args.model, window=args.window, natts=xtrain.shape[1])
        # Initialize problem
        cvoa = CVOA(size_fixed_part=3, min_size_var_part=2, max_size_var_part=11, fixed_part_max_values=[5, 8],
                    var_part_max_value=11, max_time=20,processes=args.processes, use_var_part=args.use_variable,
                    discrete=args.discrete, fitness=fitness)

    else:
        train_generator = DataGenerator(xtrain,
                                        ytrain, length=args.window,
                                        batch_size=args.batch, n_outputs=args.horizon)

        validation_generator = DataGenerator(xval,
                                             yval, length=args.window,
                                             batch_size=args.batch, n_outputs=args.horizon)

        fitness = Fitness(train_gen=train_generator, val_gen=validation_generator,
                          prediction_horizon=24, epochs=epochs, batch=batch, scaler=scaler_target,
                          model=args.model, window=args.window, natts=xtrain.shape[1])

        cvoa = CVOA(size_fixed_part=3, min_size_var_part=2, max_size_var_part=11, fixed_part_max_values=[5, 8],
                    var_part_max_value=11, max_time=20,
                    use_var_part=args.use_variable, discrete=args.discrete, fitness=fitness)

    start_time = int(round(time.time() * 1000))
    solution = cvoa.run()
    delta_time = int(round(time.time() * 1000)) - start_time

    print("Best solution: " + str(solution))
    print("Best fitness: " + str(fitness(solution)))
    print("Execution time: " + str(round(delta_time / 60000, 2)) + " mins")


# apt-get install -y libgtk2.0-dev libgtk-3-dev \
# 	libjpeg-dev libtiff-dev \
# 	libsdl1.2-dev libgstreamer-plugins-base1.0-dev \
# 	libnotify-dev freeglut3 freeglut3-dev libsm-dev \
# 	libwebkitgtk-dev libwebkitgtk-3.0-dev

if __name__ == '__main__':
    main()
