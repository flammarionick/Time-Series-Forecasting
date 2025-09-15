from tensorflow.keras import layers, models, optimizers

def build_lstm(input_shape, units1=128, units2=64, dropout=0.2, lr=1e-3, optimizer='adam', clipnorm=1.0):
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units1, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(units2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    if optimizer.lower() == 'adamw':
        opt = optimizers.AdamW(learning_rate=lr)
    elif optimizer.lower() == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=lr, clipnorm=clipnorm)
    else:
        opt = optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
    m.compile(optimizer=opt, loss='mse')
    return m