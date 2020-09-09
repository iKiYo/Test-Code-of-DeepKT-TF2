import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers


# model parmaeters
hidden_units=200 
dropout_rate=0.2"""##### train model on assist12 dataset"""
print(tf.test.is_gpu_available())

train_dataset, test_dataset, val_dataset = split_dataset(input_label_dataset, total_size=num_batches, test_fraction=0.1, val_fraction=0.2)

print(train_dataset.element_spec)
print(list(train_dataset.take(1).as_numpy_iterator()))
#print(list(test_dataset.take(1).as_numpy_iterator()))
#print(list(valid_dataset.take(1).as_numpy_iterator()))

model.predict(train_dataset.take(1))
model.summary()

"""###### train"""

print(hidden_units, dropout_rate, embed_dim, learning_rate,batch_size)
print(num_students, num_skills, max_sequence_length, num_batches)

model.fit(train_dataset.prefetch(6),  epochs=1,  validation_data=val_dataset)#,  callbacks=[CustomCallback()])

"""###### evaluate with test dataset"""

results = model.evaluate(val_dataset.prefetch(5), callbacks=[CustomCallback()])
results

print(hidden_units, dropout_rate, embed_dim, learning_rate,batch_size)
print(num_students, num_skills, max_sequence_length, num_batches)

model.fit(train_dataset.prefetch(6),  epochs=1,  validation_data=val_dataset)#,  callbacks=[CustomCallback()])
# model.fit(input_label_dataset.take(200).prefetch(1), epochs=1, callbacks=[tboard_callback])

# model.evaluate(input_label_dataset.take(1))

embed_dim = 200
learning_rate = 0.005

# build model
model = keras.Model(inputs=[x, q], outputs=outputs)


# comile model
# set Reduction.SUM for distributed traning
model.compile(loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM),
              optimizer=tf.optimizers.SGD(learning_rate=learning_rate),
                # metrics=[tf.keras.metrics.AUC()])
              metrics=[tf.keras.metrics.AUC(),tf.keras.metrics.BinaryCrossentropy()]) # keep BCEntropyfor debug

"""##### train model on assist12 dataset"""
print(tf.test.is_gpu_available())

train_dataset, test_dataset, val_dataset = split_dataset(input_label_dataset, total_size=num_batches, test_fraction=0.1, val_fraction=0.2)

print(train_dataset.element_spec)
print(list(train_dataset.take(1).as_numpy_iterator()))
#print(list(test_dataset.take(1).as_numpy_iterator()))
#print(list(valid_dataset.take(1).as_numpy_iterator()))

model.predict(train_dataset.take(1))
model.summary()

# Start trainning
print(hidden_units, dropout_rate, embed_dim, learning_rate,batch_size)
print(num_students, num_skills, max_sequence_length, num_batches)

history = model.fit(train_dataset.prefetch(6),  epochs=1,  validation_data=val_dataset)#,  callbacks=[CustomCallback()])
results = model.evaluate(val_dataset.prefetch(5))#, callbacks=[CustomCallback()])
print(results)
