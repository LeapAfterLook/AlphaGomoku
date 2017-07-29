from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


def get_model_origin():
    """
    build & compile origin model
    """

    model = Sequential()

    model.add(Conv2D(filters=1024, kernel_size=(5, 5), padding='same', input_shape=(15, 15, 1), activation='relu',
                     kernel_initializer='he_uniform', bias_initializer='he_uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                     kernel_initializer='he_uniform', bias_initializer='he_uniform'))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                     kernel_initializer='he_uniform', bias_initializer='he_uniform'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(units=1024, activation='relu', kernel_initializer='he_uniform', bias_initializer='he_uniform'))
    model.add(Dropout(0.5))

    model.add(Dense(units=225, activation='softmax', kernel_initializer='he_uniform', bias_initializer='he_uniform'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_model_padding_same():
    """
    build & compile model padding same
    """

    model = Sequential()

    model.add(Conv2D(filters=512, kernel_size=(5, 5), padding='same', input_shape=(15, 15, 1), activation='relu',
                     kernel_initializer='he_uniform', bias_initializer='he_uniform'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                     kernel_initializer='he_uniform', bias_initializer='he_uniform'))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                     kernel_initializer='he_uniform', bias_initializer='he_uniform'))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512, activation='relu', kernel_initializer='he_uniform', bias_initializer='he_uniform'))
    model.add(Dropout(0.5))

    model.add(Dense(225, activation='softmax', kernel_initializer='he_uniform', bias_initializer='he_uniform'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    model1 = get_model_origin()
    model2 = get_model_padding_same()
    model1.summary()
    model2.summary()
