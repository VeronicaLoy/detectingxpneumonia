from cnn_model import PneumoniaPrediction
import argparse

if __name__ == "__main__":
    args = {
        'epochs': 99999,
        'shuffle': True,
        'verbose': 1,
        'model_name': ''
    }
    clf = PneumoniaPrediction()
    clf.load_model('models/best_model.h5')
    clf.set_directory('data')
    clf.train(args)
    pass