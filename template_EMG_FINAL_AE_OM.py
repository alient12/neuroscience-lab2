import board
import busio
import RPi.GPIO as GPIO
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from functions_EMG_FINAL_AE_OM import *

# i2c setup
i2c = busio.I2C(board.SCL, board.SDA)           # I2C interface initialisation
ads = ADS.ADS1115(i2c)#, address=0x48)           # ADC instance, I2C communication with external ads1115 module
ads.data_rate = 860                 # maximal data rate value
ads.mode = ADS.Mode.SINGLE                 # single-shot mode for max frequency
chan1 = AnalogIn(ads, ADS.P0)         # single ended mode for chan 1
chan2 = AnalogIn(ads, ADS.P1)         # single ended mode for chan 2

# buzzer setup
BUZZER_PIN = 18               # PWM instance
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT) # set pin as output

buzzer = GPIO.PWM(BUZZER_PIN, 1000)
buzzer.start(0)

# general sampling parameters
window_size = 50                        # evaluation window length (samples)
number_window_training = 100            # number of windows to sample for training data set
number_window_testing = 100             # number of windows to sample for online testing

# EMG data acquisition for training - online mode
training_file_name = create_new_sampling_file("data/", "training_acquisition")
acquire_training_dataset(chan1, chan2, window_size, number_window_training, training_file_name)

# visual validation of data acquisition
visualize_sampling(training_file_name)

# classifier training (offline training)
classifier = train_classifier(training_file_name, window_size)

# buzzer control (online test)
testing_file_name = create_new_sampling_file("data/", "testing_acquisition")
final_labels = test_classifier(classifier, chan1, chan2, window_size, number_window_testing, buzzer, testing_file_name)
