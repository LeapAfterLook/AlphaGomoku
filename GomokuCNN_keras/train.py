from ParseGomoku.renjunet import Record, plot_board
from GomokuCNN_keras.model import get_model_origin
import numpy as np
from keras.models import load_model

# get data
x_board_images = Record.load_input_records().reshape(-1, 15, 15, 1)
y_board_labels = abs(Record.load_output_records())

x_train, x_test = x_board_images[:1000000], x_board_images[1000000:]
y_train, y_test = y_board_labels[:1000000], y_board_labels[1000000:]

# get or load model
model = get_model_origin()
# model = load_model("model_file.h5")

# train model
model.fit(x=x_train, y=y_train, batch_size=32, epochs=1, verbose=1, validation_split=0.15)

# evaluate model
# loss, accuracy = model.evaluate(x=x_test, y=y_test, batch_size=1000)

# print("Accuracy: %.2f%%" % (accuracy*100))

"""
predictions = model.predict(x=x_test, batch_size=1000)
print(predictions.shape)
# index = 400
for i in range(1000):
    predict_position = np.argmax(predictions[i])
    print(predict_position // 15, predict_position % 15)

# plot_board(x_test[index].reshape(15, 15), x_test[index + 1].reshape(15, 15))
"""

# save model
model.save('model_file.h5')
