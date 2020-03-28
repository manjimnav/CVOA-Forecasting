from CVOA.CVOA import CVOA
from ETL.ETL import *
from ETL.datagenerator import DataGenerator
from DEEP_LEARNING.LSTM import *
import time as time
import argparse
from operator import itemgetter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Path to the data", required=True)
    parser.add_argument("-g",  "--generator", help="Use generator in the data", action="store_true", default=False)
    parser.add_argument("-w", "--window", help="Window size", default=168)
    parser.add_argument("-ho", "--horizon", help="Horizon size", default=24)
    parser.add_argument("-t", "--test", help="Test size", default=.3)
    parser.add_argument("-v", "--val", help="Validation size", default=.3)
    parser.add_argument("-e", "--epochs", help="Number of epochs", default=10)
    parser.add_argument("-b", "--batch", help="Batch size", default=1024)
    parser.add_argument("-n", "--normalize", help="Normalize data", action="store_true", default=True)
    parser.add_argument("-m", "--model", help="Type of model (encoder-decoder,  encoder-attention-decoder, lstm)",
                        choices=['encoder-decoder', 'encoder-attention-decoder', 'lstm'], default='lstm')

    args = parser.parse_args()
    # Deep Learning parameters
    epochs = args.epochs
    batch = args.batch

    # Load the dataset
    data, scaler = load_data(path_to_data=args.data, useNormalization=True)
    if not args.generator:
        # Transform data to a supervised dataset
        data = data_to_supervised(data, historical_window=args.window, prediction_horizon=args.horizon)

    # Split the dataset
    xtrain, xtest, ytrain, ytest, xval, yval = splitData(data, historical_window=args.window, test_size=args.test,
                                                         val_size=args.val)

    if not args.generator:
        # Add shape to use LSTM network
        xtrain, xtest, ytrain, ytest, xval, yval = adaptShapesToLSTM(xtrain, xtest, ytrain, ytest, xval, yval)
        # Initialize problem
        cvoa = CVOA(size_fixed_part=3, min_size_var_part=2, max_size_var_part=11, fixed_part_max_values=[5, 8],
                    var_part_max_value=11, max_time=20, xtrain=xtrain, ytrain=ytrain, xval=xval, yval=yval,
                    pred_horizon=24, epochs=epochs, batch=batch, scaler=scaler, model=args.model)

    else:
        train_generator = DataGenerator(xtrain.values,
                                        ytrain.values, length=args.window,
                                        batch_size=args.batch_size, n_outputs=args.horizon)

        validation_generator = DataGenerator(xval.values,
                                             yval.values, length=args.window,
                                             batch_size=args.batch_size, n_outputs=args.horizon)

        test_generator = DataGenerator(xtest.values, ytest.values, length=args.window,
                                       batch_size=args.batch, n_outputs=args.horizon)

        cvoa = CVOA(size_fixed_part=3, min_size_var_part=2, max_size_var_part=11, fixed_part_max_values=[5, 8],
                    var_part_max_value=11, max_time=20, train_gen=train_generator, valid_gen=validation_generator,
                    pred_horizon=24, epochs=epochs, batch=batch, scaler=scaler, use_generator=args.generator,
                    model=args.model)



    time = int(round(time.time() * 1000))
    solution = cvoa.run()
    time = int(round(time.time() * 1000)) - time

    print("Best solution: " + str(solution))
    print("Best fitness: " + str(CVOA.fitness(solution)))
    print("Execution time: " + str(CVOA.df.format((time) / 60000) + " mins"))