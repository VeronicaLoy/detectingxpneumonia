from cnn_model import PneumoniaPrediction
import argparse
import logging

if __name__ == "__main__":
    # Parsing arguments...
    logging.info('Starting model...')
    clf = PneumoniaPrediction()
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--model_path', type=str)
    predict_img = parser.parse_args().img_path
    model_path = parser.parse_args().model_path
    
    # Preparing image to predict
    clf = PneumoniaPrediction()
    clf.load_model(model_path) # load trained model
    img = clf.load_img(predict_img) # load image
    prediction = clf.predict(img) # predict. returns probability
    log_info = 'Probability of Pneumonia: ' + str(round(prediction, 2))
    logging.info(log_info) # testing the prediction...
    